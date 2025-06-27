import cv2
import os
import traceback
import numpy as np
import torch
from ultralytics import YOLO
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import math
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Player:
    """Enhanced data class to represent a detected player"""
    id: int  # Local tracker ID
    global_id: Optional[int] = None  # Persistent global ID
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x1, y1, x2, y2
    center: Tuple[int, int] = (0, 0)
    confidence: float = 0.0
    class_name: str = "player"
    features: Optional[np.ndarray] = None
    color_histogram: Optional[np.ndarray] = None
    jersey_color: Optional[Tuple[int, int, int]] = None
    size_ratio: Optional[float] = None

class GlobalPlayerRegistry:
    """Enhanced with CNN embeddings and better appearance matching"""
    
    def __init__(self):
        self.next_global_id = 1
        self.global_players = {}
        self.camera_to_global = {1: {}, 2: {}}
        self.global_to_camera = {1: {}, 2: {}}
        self.feature_extractor = EnhancedFeatureExtractor()
        self.similarity_threshold = 0.35  # Fine-tuned threshold
        
        # NEW: Enhanced appearance features
        self.appearance_database = {}  # global_id -> appearance features
        self.jersey_color_clusters = {}  # Track jersey color patterns
        self.re_id_confidence_history = {}  # Track re-identification confidence

    def _extract_enhanced_appearance_features(self, player: Player, frame: np.ndarray) -> dict:
        """Extract comprehensive appearance features including CNN-like patterns"""
        x1, y1, x2, y2 = player.bbox
        roi = frame[max(0, y1):min(frame.shape[0], y2), 
                   max(0, x1):min(frame.shape[1], x2)]
        
        if roi.size == 0:
            return {}
        
        features = {}
        
        # Enhanced jersey color analysis
        jersey_color = self.feature_extractor.extract_dominant_color(roi, (0, 0, roi.shape[1], roi.shape[0]))
        features['jersey_color'] = jersey_color
        
        # Multi-scale color histograms
        features['color_hist_full'] = self.feature_extractor.extract_color_histogram(roi, (0, 0, roi.shape[1], roi.shape[0]))
        
        # Upper body color histogram (jersey focus)
        upper_roi = roi[:roi.shape[0]//2, :]  # Upper half
        if upper_roi.size > 0:
            features['color_hist_upper'] = self.feature_extractor.extract_color_histogram(upper_roi, (0, 0, upper_roi.shape[1], upper_roi.shape[0]))
        
        # Texture features using gradient patterns
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # Edge patterns
        edges = cv2.Canny(gray_roi, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        # Local Binary Pattern approximation
        sobel_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
        features['gradient_magnitude'] = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
        
        # Shape features
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            features['aspect_ratio'] = cv2.boundingRect(largest_contour)[2] / cv2.boundingRect(largest_contour)[3]
            features['contour_area_ratio'] = cv2.contourArea(largest_contour) / (roi.shape[0] * roi.shape[1])
        
        return features  
    
    def _calculate_enhanced_similarity(self, features1: dict, features2: dict) -> float:
        """Enhanced similarity calculation with multiple appearance cues"""
        similarities = []
        weights = []
        
        # Jersey color similarity (high weight)
        if 'jersey_color' in features1 and 'jersey_color' in features2:
            jersey_sim = self._calculate_jersey_similarity(features1['jersey_color'], features2['jersey_color'])
            similarities.append(jersey_sim)
            weights.append(0.4)
        
        # Color histogram similarities
        for hist_key in ['color_hist_full', 'color_hist_upper']:
            if hist_key in features1 and hist_key in features2:
                hist_sim = cv2.compareHist(
                    features1[hist_key].astype(np.float32),
                    features2[hist_key].astype(np.float32),
                    cv2.HISTCMP_CORREL
                )
                similarities.append(max(0, hist_sim))
                weights.append(0.2 if hist_key == 'color_hist_full' else 0.25)
        
        # Texture and shape similarities
        texture_features = ['edge_density', 'gradient_magnitude', 'aspect_ratio', 'contour_area_ratio']
        for feature in texture_features:
            if feature in features1 and feature in features2:
                # Normalize differences
                diff = abs(features1[feature] - features2[feature])
                max_val = max(features1[feature], features2[feature])
                sim = 1.0 - (diff / (max_val + 1e-7))
                similarities.append(max(0, sim))
                weights.append(0.05)
        
        if not similarities:
            return 0.0
        
        return np.average(similarities, weights=weights)
        
    def get_or_create_global_id(self, player: Player, camera_id: int, frame: np.ndarray) -> int:
        """FIXED: Get existing global ID or create new one for a player"""
        
        # Extract comprehensive features for the player
        player_profile = self._create_player_profile(player, frame)
        
        # First check if this local ID already has a global mapping
        if player.id in self.camera_to_global[camera_id]:
            global_id = self.camera_to_global[camera_id][player.id]
            # Update the profile with new information
            self._update_player_profile(global_id, player_profile)
            return global_id
        
        # FIXED: Only search among players NOT already mapped in THIS camera
        # This prevents multiple players from getting the same global ID
        already_mapped_global_ids = set(self.camera_to_global[camera_id].values())
        
        best_match_id = None
        best_similarity = 0.0
        
        for global_id, existing_profile in self.global_players.items():
            # Skip if this global ID is already mapped to another player in this camera
            if global_id in already_mapped_global_ids:
                continue
                
            similarity = self._calculate_profile_similarity(player_profile, existing_profile)
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match_id = global_id
        
        if best_match_id is not None:
            # Found existing player - create mapping
            self._create_mapping(player.id, best_match_id, camera_id)
            self._update_player_profile(best_match_id, player_profile)
            return best_match_id
        else:
            # Create new global player
            global_id = self.next_global_id
            self.next_global_id += 1
            self.global_players[global_id] = player_profile
            self._create_mapping(player.id, global_id, camera_id)
            return global_id

    
    def _create_player_profile(self, player: Player, frame: np.ndarray) -> dict:
        """Create comprehensive player profile for matching"""
        profile = {
            'jersey_color': player.jersey_color,
            'color_histogram': player.color_histogram.copy() if player.color_histogram is not None else None,
            'bbox_history': [player.bbox],
            'size_history': [(player.bbox[2] - player.bbox[0], player.bbox[3] - player.bbox[1])],
            'confidence_history': [player.confidence],
            'last_seen_frame': 0,
            'appearances': 1
        }
        return profile
    
    def _update_player_profile(self, global_id: int, new_profile: dict):
        """Update existing player profile with new information"""
        if global_id not in self.global_players:
            return
        
        profile = self.global_players[global_id]
        
        # Update histories (keep last N entries)
        max_history = 10
        profile['bbox_history'].append(new_profile['bbox_history'][0])
        profile['size_history'].append(new_profile['size_history'][0])
        profile['confidence_history'].append(new_profile['confidence_history'][0])
        
        # Trim histories
        profile['bbox_history'] = profile['bbox_history'][-max_history:]
        profile['size_history'] = profile['size_history'][-max_history:]
        profile['confidence_history'] = profile['confidence_history'][-max_history:]
        
        # Update other fields
        profile['last_seen_frame'] = new_profile['last_seen_frame']
        profile['appearances'] += 1
        
        # Update jersey color if more confident
        if (new_profile['jersey_color'] is not None and 
            new_profile['confidence_history'][0] > np.mean(profile['confidence_history'])):
            profile['jersey_color'] = new_profile['jersey_color']
        
        # Update color histogram (running average)
        if new_profile['color_histogram'] is not None and profile['color_histogram'] is not None:
            alpha = 0.3  # Learning rate
            profile['color_histogram'] = (1 - alpha) * profile['color_histogram'] + alpha * new_profile['color_histogram']
    
    def _calculate_profile_similarity(self, profile1: dict, profile2: dict) -> float:
        """Calculate similarity between two player profiles"""
        similarities = []
        weights = []
        
        # Jersey color similarity
        if profile1['jersey_color'] is not None and profile2['jersey_color'] is not None:
            jersey_sim = self._calculate_jersey_similarity(profile1['jersey_color'], profile2['jersey_color'])
            similarities.append(jersey_sim)
            weights.append(0.4)
        
        # Color histogram similarity
        if profile1['color_histogram'] is not None and profile2['color_histogram'] is not None:
            hist_sim = cv2.compareHist(
                profile1['color_histogram'].astype(np.float32),
                profile2['color_histogram'].astype(np.float32),
                cv2.HISTCMP_CORREL
            )
            similarities.append(max(0, hist_sim))
            weights.append(0.4)
        
        # Size consistency
        avg_size1 = np.mean(profile1['size_history'], axis=0)
        avg_size2 = np.mean(profile2['size_history'], axis=0)
        size_ratio = min(avg_size1[0] * avg_size1[1], avg_size2[0] * avg_size2[1]) / max(avg_size1[0] * avg_size1[1], avg_size2[0] * avg_size2[1])
        similarities.append(size_ratio)
        weights.append(0.2)
        
        if not similarities:
            return 0.0
        
        return np.average(similarities, weights=weights)
    
    def _calculate_jersey_similarity(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        """Calculate jersey color similarity in LAB space"""
        try:
            lab1 = cv2.cvtColor(np.uint8([[color1]]), cv2.COLOR_BGR2LAB)[0][0]
            lab2 = cv2.cvtColor(np.uint8([[color2]]), cv2.COLOR_BGR2LAB)[0][0]
            distance = np.sqrt(np.sum((lab1.astype(float) - lab2.astype(float)) ** 2))
            max_distance = 100 * np.sqrt(3)
            return max(0, 1.0 - (distance / max_distance))
        except:
            return 0.0
    
    def _create_mapping(self, local_id: int, global_id: int, camera_id: int):
        """Create bidirectional mapping between local and global IDs"""
        self.camera_to_global[camera_id][local_id] = global_id
        self.global_to_camera[camera_id][global_id] = local_id
    
    def get_global_id(self, local_id: int, camera_id: int) -> Optional[int]:
        """Get global ID for a local camera ID"""
        return self.camera_to_global[camera_id].get(local_id)
    
    def cleanup_old_mappings(self, active_local_ids_cam1: set, active_local_ids_cam2: set):
        """Clean up mappings for players no longer detected"""
        for camera_id, active_ids in [(1, active_local_ids_cam1), (2, active_local_ids_cam2)]:
            # Find local IDs to remove
            local_ids_to_remove = []
            for local_id in self.camera_to_global[camera_id]:
                if local_id not in active_ids:
                    local_ids_to_remove.append(local_id)
            
            # Remove mappings
            for local_id in local_ids_to_remove:
                global_id = self.camera_to_global[camera_id][local_id]
                del self.camera_to_global[camera_id][local_id]
                if global_id in self.global_to_camera[camera_id]:
                    del self.global_to_camera[camera_id][global_id]

class EnhancedFeatureExtractor:
    """Enhanced feature extraction for better player matching"""
    
    def __init__(self):
        self.color_bins = 32
        self.hog = cv2.HOGDescriptor()
    
    def extract_dominant_color(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
        """Extract dominant color (likely jersey color) from upper body region"""
        x1, y1, x2, y2 = bbox
        # Focus on upper 60% of the bounding box for jersey color
        upper_y2 = y1 + int((y2 - y1) * 0.6)
        roi = image[max(0, y1):min(image.shape[0], upper_y2), 
                   max(0, x1):min(image.shape[1], x2)]
        
        if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
            return (128, 128, 128)  # Default gray
        
        try:
            # Reshape for k-means clustering
            roi_reshaped = roi.reshape(-1, 3)
            roi_reshaped = np.float32(roi_reshaped)
            
            # K-means to find dominant colors
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            k = min(3, len(roi_reshaped))  # Ensure k is not larger than data points
            
            if k < 1:
                return (128, 128, 128)
                
            _, labels, centers = cv2.kmeans(roi_reshaped, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Return the most frequent cluster center
            unique, counts = np.unique(labels, return_counts=True)
            dominant_color_idx = unique[np.argmax(counts)]
            dominant_color = centers[dominant_color_idx]
            
            return tuple(map(int, dominant_color))
        except:
            return (128, 128, 128)  # Default gray on error
    
    def extract_color_histogram(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract enhanced color histogram with better normalization"""
        x1, y1, x2, y2 = bbox
        roi = image[max(0, y1):min(image.shape[0], y2), 
                   max(0, x1):min(image.shape[1], x2)]
        
        if roi.size == 0:
            return np.zeros((self.color_bins * 3,))
        
        try:
            # Convert to HSV for better color representation
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Calculate histogram for each channel with better ranges
            hist_h = cv2.calcHist([hsv], [0], None, [self.color_bins], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [self.color_bins], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [self.color_bins], [0, 256])
            
            # Normalize each histogram separately to prevent dominance
            hist_h = hist_h.flatten() / (np.sum(hist_h) + 1e-7)
            hist_s = hist_s.flatten() / (np.sum(hist_s) + 1e-7)
            hist_v = hist_v.flatten() / (np.sum(hist_v) + 1e-7)
            
            # Concatenate with weights (hue is most important for jersey matching)
            hist = np.concatenate([hist_h * 2.0, hist_s * 1.5, hist_v * 1.0])
            
            return hist
        except:
            return np.zeros((self.color_bins * 3,))
    
    def extract_spatial_features(self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> np.ndarray:
        """Extract enhanced spatial features"""
        x1, y1, x2, y2 = bbox
        height, width = frame_shape[:2]
        
        # Basic normalized coordinates
        center_x = (x1 + x2) / (2 * width)
        center_y = (y1 + y2) / (2 * height)
        bbox_width = (x2 - x1) / width
        bbox_height = (y2 - y1) / height
        aspect_ratio = bbox_width / (bbox_height + 1e-7)
        
        # Additional features
        area_ratio = (bbox_width * bbox_height)
        bottom_y = y2 / height  # Important for field position
        
        return np.array([center_x, center_y, bbox_width, bbox_height, aspect_ratio, area_ratio, bottom_y])

class EnhancedPlayerMatcher:
    """Enhanced player matching with multiple algorithms"""
    
    def __init__(self, color_weight: float = 0.5, spatial_weight: float = 0.3, 
                 jersey_weight: float = 0.2):
        self.color_weight = color_weight
        self.spatial_weight = spatial_weight
        self.jersey_weight = jersey_weight
        self.feature_extractor = EnhancedFeatureExtractor()
        self.player_history = defaultdict(list)
        self.max_history = 5
    
    def calculate_color_similarity(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Enhanced color similarity using multiple metrics"""
        if hist1 is None or hist2 is None or len(hist1) == 0 or len(hist2) == 0:
            return 0.0
        
        try:
            # Correlation coefficient
            correlation = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)
            
            # Chi-square distance (inverted to similarity)
            chi_square = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CHISQR)
            chi_square_sim = 1.0 / (1.0 + chi_square)
            
            # Bhattacharyya distance (inverted to similarity)
            bhattacharyya = cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
            bhattacharyya_sim = 1.0 - bhattacharyya
            
            # Combined similarity
            combined_sim = (correlation * 0.5 + chi_square_sim * 0.3 + bhattacharyya_sim * 0.2)
            return max(0, combined_sim)
        except:
            return 0.0
    
    def calculate_jersey_similarity(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        """Calculate jersey color similarity"""
        if color1 is None or color2 is None:
            return 0.0
        
        try:
            # Convert to LAB color space for perceptual distance
            lab1 = cv2.cvtColor(np.uint8([[color1]]), cv2.COLOR_BGR2LAB)[0][0]
            lab2 = cv2.cvtColor(np.uint8([[color2]]), cv2.COLOR_BGR2LAB)[0][0]
            
            # Calculate Euclidean distance in LAB space
            distance = np.sqrt(np.sum((lab1.astype(float) - lab2.astype(float)) ** 2))
            
            # Convert to similarity (max distance in LAB is ~100*sqrt(3))
            max_distance = 100 * np.sqrt(3)
            similarity = 1.0 - (distance / max_distance)
            
            return max(0, similarity)
        except:
            return 0.0
    
    def calculate_spatial_similarity(self, spatial1: np.ndarray, spatial2: np.ndarray) -> float:
        """Enhanced spatial similarity calculation"""
        try:
            diff = np.abs(spatial1 - spatial2)
            
            # Different weights for different features
            # [center_x, center_y, width, height, aspect_ratio, area_ratio, bottom_y]
            weights = np.array([1.0, 1.2, 0.8, 0.8, 0.5, 0.6, 1.1])
            weighted_diff = np.sum(diff * weights)
            
            # Exponential decay with adjusted scaling
            return np.exp(-weighted_diff * 3)
        except:
            return 0.0
    
    def calculate_similarity(self, player1: Player, player2: Player, 
                       frame_shape1: Tuple[int, int], frame_shape2: Tuple[int, int]) -> float:
        """FIXED: Calculate comprehensive similarity with better weighting"""
        similarities = []
        weights = []
        
        # Color histogram similarity
        if player1.color_histogram is not None and player2.color_histogram is not None:
            color_sim = self.calculate_color_similarity(player1.color_histogram, player2.color_histogram)
            similarities.append(color_sim)
            weights.append(self.color_weight)
        
        # Jersey color similarity
        if player1.jersey_color is not None and player2.jersey_color is not None:
            jersey_sim = self.calculate_jersey_similarity(player1.jersey_color, player2.jersey_color)
            similarities.append(jersey_sim)
            weights.append(self.jersey_weight)
        
        # FIXED: More sophisticated spatial similarity for cross-camera matching
        spatial1 = self.feature_extractor.extract_spatial_features(player1.bbox, frame_shape1)
        spatial2 = self.feature_extractor.extract_spatial_features(player2.bbox, frame_shape2)
        
        # FIXED: Cross-camera spatial matching (less strict on position, more on relative features)
        spatial_sim = self.calculate_cross_camera_spatial_similarity(spatial1, spatial2)
        similarities.append(spatial_sim)
        weights.append(self.spatial_weight)
        
        if not similarities:
            return 0.0
        
        # FIXED: Use minimum similarity boost for very low matches
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        
        weighted_sim = sum(s * w for s, w in zip(similarities, weights)) / total_weight
        
        # FIXED: Add bonus for multiple feature agreement
        if len(similarities) >= 2:
            # Bonus if multiple features agree (all above certain threshold)
            agreement_threshold = 0.3
            agreements = sum(1 for s in similarities if s > agreement_threshold)
            if agreements >= 2:
                weighted_sim *= (1.0 + 0.1 * agreements)  # Up to 30% bonus
        
        return min(1.0, weighted_sim)  # Ensure it doesn't exceed 1.0
    
    def calculate_cross_camera_spatial_similarity(self, spatial1: np.ndarray, spatial2: np.ndarray) -> float:
        """FIXED: New method for cross-camera spatial comparison"""
        try:
            # For cross-camera matching, focus on relative size and proportions rather than absolute position
            # [center_x, center_y, width, height, aspect_ratio, area_ratio, bottom_y]
            
            # Compare aspect ratios (should be similar across cameras)
            aspect_diff = abs(spatial1[4] - spatial2[4])
            aspect_sim = max(0, 1.0 - aspect_diff * 2)  # Penalize aspect ratio differences
            
            # Compare relative sizes (should be reasonably similar)
            size_diff = abs(spatial1[5] - spatial2[5])  # area_ratio
            size_sim = max(0, 1.0 - size_diff * 3)
            
            # Compare relative vertical positions (less important for cross-camera)
            vertical_diff = abs(spatial1[6] - spatial2[6])  # bottom_y
            vertical_sim = max(0, 1.0 - vertical_diff * 1.5)
            
            # Weighted combination (emphasize aspect ratio and size)
            cross_camera_sim = (aspect_sim * 0.5 + size_sim * 0.4 + vertical_sim * 0.1)
            
            return cross_camera_sim
        except:
            return 0.0
    
    def match_players_hungarian(self, players_cam1: List[Player], players_cam2: List[Player],
                           frame_shape1: Tuple[int, int], frame_shape2: Tuple[int, int]) -> Dict[int, int]:
        """FIXED: Hungarian algorithm with better threshold and fallback"""
        if not players_cam1 or not players_cam2:
            return {}
        
        # Calculate similarity matrix
        similarity_matrix = np.zeros((len(players_cam1), len(players_cam2)))
        
        for i, p1 in enumerate(players_cam1):
            for j, p2 in enumerate(players_cam2):
                similarity_matrix[i, j] = self.calculate_similarity(p1, p2, frame_shape1, frame_shape2)
        
        # FIXED: Debug output for similarity matrix
        print(f"Similarity matrix max: {np.max(similarity_matrix):.3f}, mean: {np.mean(similarity_matrix):.3f}")
        
        # Convert similarity to cost matrix (Hungarian minimizes cost)
        cost_matrix = 1.0 - similarity_matrix
        
        # Apply Hungarian algorithm
        try:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # FIXED: Use adaptive threshold based on actual similarities found
            base_threshold = 0.15  # Lower base threshold
            max_similarity = np.max(similarity_matrix)
            
            if max_similarity > 0.5:
                similarity_threshold = 0.25  # Higher threshold if good matches exist
            elif max_similarity > 0.3:
                similarity_threshold = 0.20  # Medium threshold
            else:
                similarity_threshold = base_threshold  # Low threshold for difficult cases
            
            print(f"Using similarity threshold: {similarity_threshold:.3f}")
            
            # Create matches dictionary
            matches = {}
            for i, j in zip(row_indices, col_indices):
                if similarity_matrix[i, j] > similarity_threshold:
                    matches[players_cam1[i].id] = players_cam2[j].id
                    print(f"Match: Player {players_cam1[i].id} ↔ Player {players_cam2[j].id} (sim: {similarity_matrix[i, j]:.3f})")
            
            return matches
            
        except Exception as e:
            print(f"Hungarian algorithm failed: {e}")
            return {}
        
# new class for tracking players across frames
class PlayerTracker:
    """Tracks players across frames to maintain consistent IDs with smooth transitions"""
    
    def __init__(self, max_disappeared: int = 15, max_distance: float = 200.0):  # Increased buffer
        self.next_object_id = 1
        self.objects = {}  # Store active players {id: Player}
        self.disappeared = {}  # Track how many frames a player has been missing
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.feature_extractor = EnhancedFeatureExtractor()
        
        # NEW: Occlusion buffering and ID stability
        self.occlusion_buffer = {}  # Store players during brief occlusions
        self.id_stability_tracker = {}  # Track ID assignment confidence over time
        self.temporal_features = {}  # Store temporal feature patterns
        self.id_swap_penalty = 0.3  # Penalty for changing IDs

    def _calculate_temporal_consistency(self, player_id: int, new_features: dict) -> float:
        """Calculate how consistent new features are with player's history"""
        if player_id not in self.temporal_features:
            return 0.5  # Neutral score for new players
        
        history = self.temporal_features[player_id]
        consistency_scores = []
        
        # Check jersey color consistency
        if 'jersey_colors' in history and new_features.get('jersey_color'):
            recent_colors = history['jersey_colors'][-5:]  # Last 5 colors
            color_consistency = np.mean([
                self._calculate_jersey_similarity(new_features['jersey_color'], color)
                for color in recent_colors
            ])
            consistency_scores.append(color_consistency)
        
        # Check size consistency
        if 'sizes' in history and new_features.get('size'):
            recent_sizes = history['sizes'][-5:]
            avg_size = np.mean(recent_sizes)
            size_consistency = 1.0 - abs(new_features['size'] - avg_size) / avg_size
            consistency_scores.append(max(0, size_consistency))
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _update_temporal_features(self, player_id: int, features: dict):
        """Update temporal feature history for a player"""
        if player_id not in self.temporal_features:
            self.temporal_features[player_id] = {
                'jersey_colors': [],
                'sizes': [],
                'positions': []
            }
        
        history = self.temporal_features[player_id]
        
        if features.get('jersey_color'):
            history['jersey_colors'].append(features['jersey_color'])
            history['jersey_colors'] = history['jersey_colors'][-10:]  # Keep last 10
        
        if features.get('size'):
            history['sizes'].append(features['size'])
            history['sizes'] = history['sizes'][-10:]
        
        if features.get('position'):
            history['positions'].append(features['position'])
            history['positions'] = history['positions'][-10:]
    
    def _calculate_tracking_similarity(self, player1: Player, player2: Player, frame_shape: Tuple[int, int]) -> float:
        """FIXED: Calculate similarity for tracking with better thresholds"""
        # Position similarity (most important for short-term tracking)
        center1 = np.array(player1.center)
        center2 = np.array(player2.center)
        distance = np.linalg.norm(center1 - center2)
        
        # FIXED: Use adaptive distance threshold based on player size
        player_size = np.sqrt((player1.bbox[2] - player1.bbox[0]) * (player1.bbox[3] - player1.bbox[1]))
        adaptive_max_distance = max(self.max_distance, player_size * 2)  # Allow movement up to 2x player size
        
        position_sim = max(0, 1.0 - (distance / adaptive_max_distance))
        
        # FIXED: More lenient size similarity
        area1 = (player1.bbox[2] - player1.bbox[0]) * (player1.bbox[3] - player1.bbox[1])
        area2 = (player2.bbox[2] - player2.bbox[0]) * (player2.bbox[3] - player2.bbox[1])
        size_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
        
        # FIXED: Apply minimum size ratio threshold to prevent extreme size differences
        if size_ratio < 0.3:  # If size difference is too extreme
            size_ratio = 0.1  # Heavily penalize but don't completely exclude
        
        # Color similarity (if available)
        color_sim = 0.5  # Default neutral value
        if player1.color_histogram is not None and player2.color_histogram is not None:
            try:
                color_sim = cv2.compareHist(
                    player1.color_histogram.astype(np.float32), 
                    player2.color_histogram.astype(np.float32), 
                    cv2.HISTCMP_CORREL
                )
                color_sim = max(0, color_sim)
            except:
                color_sim = 0.5
        
        # FIXED: Adjusted weights - prioritize position more, be more lenient overall
        total_sim = (position_sim * 0.7 + size_ratio * 0.15 + color_sim * 0.15)
        return total_sim
    
    def _calculate_jersey_similarity(self, color1, color2) -> float:
        """Calculate similarity between two jersey colors"""
        if color1 is None or color2 is None:
            return 0.0
        
        try:
            # Convert colors to numpy arrays if they aren't already
            c1 = np.array(color1, dtype=np.float32)
            c2 = np.array(color2, dtype=np.float32)
            
            # Calculate Euclidean distance in BGR color space
            distance = np.linalg.norm(c1 - c2)
            
            # Convert distance to similarity (0-1 range)
            # Maximum possible distance in BGR space is sqrt(255^2 * 3) ≈ 441
            max_distance = 441.67
            similarity = max(0, 1.0 - (distance / max_distance))
            
            return similarity
            
        except Exception as e:
            print(f"Error calculating jersey color similarity: {e}")
            return 0.0

    def _fallback_tracking(self, detections: List[Player], frame: np.ndarray) -> List[Player]:
        """Fallback tracking method when enhanced tracking fails"""
        tracked_players = []
        
        for detection in detections:
            # Simple assignment of new IDs
            detection.id = self.next_object_id
            self.objects[self.next_object_id] = detection
            self.next_object_id += 1
            tracked_players.append(detection)
        
        return tracked_players

    def _cleanup_buffers_and_disappeared(self):
        """Clean up tracking buffers and disappeared objects"""
        # Clean up occlusion buffer
        to_remove_from_buffer = []
        for obj_id, buffer_data in self.occlusion_buffer.items():
            if buffer_data['frames_missing'] > self.max_disappeared:
                to_remove_from_buffer.append(obj_id)
        
        for obj_id in to_remove_from_buffer:
            del self.occlusion_buffer[obj_id]
            if obj_id in self.objects:
                del self.objects[obj_id]
            if obj_id in self.temporal_features:
                del self.temporal_features[obj_id]
        
        # Clean up disappeared objects
        to_remove_disappeared = []
        for obj_id, frames_disappeared in self.disappeared.items():
            if frames_disappeared > self.max_disappeared:
                to_remove_disappeared.append(obj_id)
        
        for obj_id in to_remove_disappeared:
            del self.disappeared[obj_id]
            if obj_id in self.objects:
                del self.objects[obj_id]
            if obj_id in self.temporal_features:
                del self.temporal_features[obj_id]
            if obj_id in self.id_stability_tracker:
                del self.id_stability_tracker[obj_id]
    
    def update(self, detections: List[Player], frame: np.ndarray) -> List[Player]:
        """Enhanced update with smooth ID transitions and occlusion handling"""
        tracked_players = []
        
        # Extract features for all detections
        for detection in detections:
            detection.jersey_color = self.feature_extractor.extract_dominant_color(
                frame, detection.bbox)
            detection.color_histogram = self.feature_extractor.extract_color_histogram(
                frame, detection.bbox)
            # Add size feature
            detection.size = (detection.bbox[2] - detection.bbox[0]) * (detection.bbox[3] - detection.bbox[1])
        
        if not self.objects:
            # Initialize new objects
            for detection in detections:
                detection.id = self.next_object_id
                self.objects[self.next_object_id] = detection
                self._update_temporal_features(self.next_object_id, {
                    'jersey_color': detection.jersey_color,
                    'size': detection.size,
                    'position': detection.center
                })
                self.next_object_id += 1
                tracked_players.append(detection)
            return tracked_players
        
        # Enhanced similarity calculation with temporal consistency
        object_ids = list(self.objects.keys())
        similarity_matrix = np.zeros((len(object_ids), len(detections)))
        
        for i, obj_id in enumerate(object_ids):
            for j, detection in enumerate(detections):
                # Base similarity
                base_similarity = self._calculate_tracking_similarity(
                    self.objects[obj_id], detection, frame.shape)
                
                # Temporal consistency bonus
                temporal_features = {
                    'jersey_color': detection.jersey_color,
                    'size': detection.size,
                    'position': detection.center
                }
                temporal_consistency = self._calculate_temporal_consistency(obj_id, temporal_features)
                
                # ID stability bonus (prefer keeping existing assignments)
                stability_bonus = self.id_stability_tracker.get(obj_id, 0) * 0.1
                
                # Combined similarity
                similarity_matrix[i, j] = (base_similarity * 0.7 + 
                                         temporal_consistency * 0.2 + 
                                         stability_bonus * 0.1)
        
        # Hungarian assignment with ID swap penalty
        try:
            cost_matrix = 1.0 - similarity_matrix
            # Add ID swap penalty to cost matrix
            for i in range(len(object_ids)):
                for j in range(len(detections)):
                    if similarity_matrix[i, j] < 0.8:  # Only penalize uncertain matches
                        cost_matrix[i, j] += self.id_swap_penalty
            
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Process assignments with improved threshold
            similarity_threshold = 0.25
            used_detection_indices = set()
            used_object_indices = set()
            
            for i, j in zip(row_indices, col_indices):
                if similarity_matrix[i, j] > similarity_threshold:
                    obj_id = object_ids[i]
                    detection = detections[j]
                    detection.id = obj_id
                    
                    # Update object and temporal features
                    self.objects[obj_id] = detection
                    self._update_temporal_features(obj_id, {
                        'jersey_color': detection.jersey_color,
                        'size': detection.size,
                        'position': detection.center
                    })
                    
                    # Increase ID stability
                    self.id_stability_tracker[obj_id] = self.id_stability_tracker.get(obj_id, 0) + 1
                    
                    if obj_id in self.disappeared:
                        del self.disappeared[obj_id]
                    
                    tracked_players.append(detection)
                    used_detection_indices.add(j)
                    used_object_indices.add(i)
            
            # Handle unmatched detections
            for j, detection in enumerate(detections):
                if j not in used_detection_indices:
                    detection.id = self.next_object_id
                    self.objects[self.next_object_id] = detection
                    self._update_temporal_features(self.next_object_id, {
                        'jersey_color': detection.jersey_color,
                        'size': detection.size,
                        'position': detection.center
                    })
                    self.next_object_id += 1
                    tracked_players.append(detection)
            
            # Enhanced occlusion handling for disappeared objects
            for i, obj_id in enumerate(object_ids):
                if i not in used_object_indices:
                    # Move to occlusion buffer instead of immediate disappearance
                    if obj_id not in self.occlusion_buffer:
                        self.occlusion_buffer[obj_id] = {
                            'player': self.objects[obj_id],
                            'frames_missing': 1,
                            'last_position': self.objects[obj_id].center
                        }
                    else:
                        self.occlusion_buffer[obj_id]['frames_missing'] += 1
                    
                    # Only mark as disappeared after buffer period
                    if self.occlusion_buffer[obj_id]['frames_missing'] > 5:  # Buffer 5 frames
                        if obj_id not in self.disappeared:
                            self.disappeared[obj_id] = 1
                        else:
                            self.disappeared[obj_id] += 1
        
        except Exception as e:
            print(f"Enhanced tracking failed: {e}")
            # Fallback to simple tracking
            return self._fallback_tracking(detections, frame)
        
        # Cleanup
        self._cleanup_buffers_and_disappeared()
        
        return tracked_players
    
class EnhancedCrossCameraPlayerMapper:
    def __init__(self, model_path: str = None):
        # Load YOLO model with better error handling
        if model_path and os.path.exists(model_path):
            print(f"Loading custom model: {model_path}")
            self.model = YOLO(model_path)
            print(f"Model classes: {self.model.names}")
            print(f"Number of classes: {len(self.model.names)}")
        else:
            print("Loading default YOLOv8 model...")
            self.model = YOLO('yolov8n.pt')
        
        # Test model on a sample frame
        self._test_model_functionality()
        
        self.feature_extractor = EnhancedFeatureExtractor()
        self.player_matcher = EnhancedPlayerMatcher()
        
        self.tracker_cam1 = PlayerTracker(max_disappeared=10, max_distance=150.0)
        self.tracker_cam2 = PlayerTracker(max_disappeared=10, max_distance=150.0)
        
        # ADD GLOBAL REGISTRY
        self.global_registry = GlobalPlayerRegistry()

        # Detection parameters
        self.confidence_threshold = 0.25
        self.player_class_ids = [1, 2]  # goalkeeper and player classes
        self.referee_class_id = 3
    
    def _test_model_functionality(self):
        """Test model with a dummy image"""
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            results = self.model(dummy_image, verbose=False)
            print("Model test successful")
        except Exception as e:
            print(f"Model test failed: {e}")
    
    def detect_players_raw(self, frame: np.ndarray) -> List[Player]:
        """Raw detection without ID assignment (for internal use)"""
        try:
            results = self.model(frame, conf=self.confidence_threshold, iou=0.5, verbose=False)
            
            raw_detections = []
            temp_id = 0
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names.get(class_id, f"class_{class_id}")
                        
                        if class_id in self.player_class_ids:
                            bbox_width = x2 - x1
                            bbox_height = y2 - y1
                            
                            if bbox_width > 15 and bbox_height > 30:
                                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                                
                                player = Player(
                                    id=temp_id,  # Temporary ID, will be assigned by tracker
                                    bbox=(x1, y1, x2, y2),
                                    center=center,
                                    confidence=confidence,
                                    class_name=class_name
                                )
                                
                                raw_detections.append(player)
                                temp_id += 1
            
            return raw_detections
            
        except Exception as e:
            print(f"Detection failed: {e}")
            traceback.print_exc()
            return []
    
    def detect_players(self, frame: np.ndarray, camera_id: int) -> List[Player]:
        """UPDATED: Detect players with global ID assignment"""
        # Get raw detections
        raw_detections = self.detect_players_raw(frame)
        
        # Use appropriate tracker based on camera
        if camera_id == 1:
            tracked_players = self.tracker_cam1.update(raw_detections, frame)
        else:
            tracked_players = self.tracker_cam2.update(raw_detections, frame)
        
        # ASSIGN GLOBAL IDs
        for player in tracked_players:
            global_id = self.global_registry.get_or_create_global_id(player, camera_id, frame)
            player.global_id = global_id
        
        return tracked_players
    
    
    def process_frames(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[List[Player], List[Player], Dict[int, int]]:
        """UPDATED: Process frames with global ID management"""
        try:
            # Detect players with global IDs
            players_cam1 = self.detect_players(frame1, camera_id=1)
            players_cam2 = self.detect_players(frame2, camera_id=2)
            
            print(f"Detected {len(players_cam1)} players in cam1 (Global IDs: {[p.global_id for p in players_cam1]})")
            print(f"Detected {len(players_cam2)} players in cam2 (Global IDs: {[p.global_id for p in players_cam2]})")
            
            # Clean up old mappings
            active_local_ids_cam1 = {p.id for p in players_cam1}
            active_local_ids_cam2 = {p.id for p in players_cam2}
            self.global_registry.cleanup_old_mappings(active_local_ids_cam1, active_local_ids_cam2)
            
            # Create cross-camera mappings based on global IDs
            current_mappings = {}
            global_ids_cam1 = {p.global_id for p in players_cam1 if p.global_id is not None}
            global_ids_cam2 = {p.global_id for p in players_cam2 if p.global_id is not None}
            
            # Find players that appear in both cameras
            common_global_ids = global_ids_cam1.intersection(global_ids_cam2)
            
            for global_id in common_global_ids:
                # Find local IDs for this global ID
                cam1_player = next((p for p in players_cam1 if p.global_id == global_id), None)
                cam2_player = next((p for p in players_cam2 if p.global_id == global_id), None)
                
                if cam1_player and cam2_player:
                    current_mappings[cam1_player.id] = cam2_player.id
            
            print(f"Cross-camera mappings (local IDs): {current_mappings}")
            print(f"Common global IDs: {common_global_ids}")
            
            return players_cam1, players_cam2, current_mappings
            
        except Exception as e:
            print(f"Error in process_frames: {e}")
            traceback.print_exc()
            return [], [], {}
    
    
    def draw_detections(self, frame: np.ndarray, players: List[Player], 
                   mappings: Dict[int, int] = None, is_cam1: bool = True) -> np.ndarray:
        """Enhanced visualization with persistent IDs and trajectory trails"""
        output_frame = frame.copy()
        
        # Draw trajectory trails (if position history available)
        for player in players:
            if hasattr(player, 'position_history') and len(player.position_history) > 1:
                points = np.array(player.position_history[-10:], np.int32)  # Last 10 positions
                for i in range(1, len(points)):
                    alpha = i / len(points)  # Fade trail
                    color = tuple(int(c * alpha) for c in (255, 255, 255))
                    cv2.line(output_frame, tuple(points[i-1]), tuple(points[i]), color, 2)
        
        for player in players:
            x1, y1, x2, y2 = player.bbox
            
            # Enhanced color coding based on ID stability and global mapping
            if player.global_id is not None:
                # Use persistent color based on global ID
                persistent_color = self._get_persistent_color(player.global_id)
                
                if mappings and player.id in mappings:
                    color = persistent_color  # Full color for matched players
                    mapped_id = mappings[player.id]
                    stability_score = self.tracker_cam1.id_stability_tracker.get(player.id, 0) if is_cam1 else self.tracker_cam2.id_stability_tracker.get(player.id, 0)
                    label = f"G{player.global_id} (L{player.id}↔L{mapped_id}) S:{stability_score}"
                else:
                    # Dim the color for unmatched but tracked players
                    color = tuple(int(c * 0.7) for c in persistent_color)
                    stability_score = self.tracker_cam1.id_stability_tracker.get(player.id, 0) if is_cam1 else self.tracker_cam2.id_stability_tracker.get(player.id, 0)
                    label = f"G{player.global_id} (L{player.id}) S:{stability_score}"
            else:
                color = (128, 128, 128)  # Gray for untracked
                label = f"L{player.id} (new)"
            
            # Enhanced bounding box with thickness based on confidence
            thickness = max(1, int(player.confidence * 4))
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw jersey color patch with improved visibility
            if player.jersey_color:
                jersey_color_bgr = player.jersey_color
                patch_size = 25
                cv2.rectangle(output_frame, (x2 - patch_size, y1), (x2, y1 + patch_size), 
                            jersey_color_bgr, -1)
                cv2.rectangle(output_frame, (x2 - patch_size, y1), (x2, y1 + patch_size), 
                            (0, 0, 0), 2)
            
            # Enhanced label with background and better positioning
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            label_bg_color = tuple(int(c * 0.8) for c in color)
            cv2.rectangle(output_frame, (x1, y1 - 30), (x1 + label_size[0] + 5, y1), 
                        label_bg_color, -1)
            cv2.putText(output_frame, label, (x1 + 2, y1 - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Draw confidence and additional info
            info_text = f"C:{player.confidence:.2f}"
            if hasattr(player, 'temporal_consistency'):
                info_text += f" T:{player.temporal_consistency:.2f}"
            
            cv2.putText(output_frame, info_text, (x1, y2 + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw center point
            cv2.circle(output_frame, player.center, 3, color, -1)
        
        return output_frame
    
    def _get_persistent_color(self, global_id: int) -> Tuple[int, int, int]:
        """Generate consistent color for each global ID"""
        # Use global ID to generate consistent color
        np.random.seed(global_id)
        hue = (global_id * 137) % 360  # Golden angle for good color distribution
        saturation = 200
        value = 255
        
        # Convert HSV to BGR
        hsv = np.uint8([[[hue//2, saturation, value]]])  # OpenCV uses H in 0-180 range
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        return tuple(map(int, bgr))
    
    def analyze_detection_quality(self, frame: np.ndarray) -> Dict:
        """Fixed detection analysis"""
        try:
            results = self.model(frame, conf=0.01, iou=0.9, verbose=False)
            
            analysis = {
                'total_detections': 0,
                'player_detections': 0,
                'goalkeeper_detections': 0,
                'referee_detections': 0,
                'filtered_players': 0,
                'detection_details': [],
                'classes_found': set(),
                'confidence_range': [1.0, 0.0]
            }
            
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    analysis['total_detections'] = len(boxes)
                    
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names.get(class_id, f"unknown_{class_id}")
                        
                        analysis['classes_found'].add(class_name)
                        analysis['confidence_range'][0] = min(analysis['confidence_range'][0], confidence)
                        analysis['confidence_range'][1] = max(analysis['confidence_range'][1], confidence)
                        
                        # Count by class
                        if class_id == 2:  # player
                            analysis['player_detections'] += 1
                        elif class_id == 1:  # goalkeeper
                            analysis['goalkeeper_detections'] += 1
                        elif class_id == 3:  # referee
                            analysis['referee_detections'] += 1
                        
                        # Check if would be filtered
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        if (class_id in self.player_class_ids and 
                            bbox_width > 15 and bbox_height > 30 and 
                            confidence > self.confidence_threshold):
                            analysis['filtered_players'] += 1
                        
                        detail = {
                            'class': class_name,
                            'class_id': class_id,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2],
                            'area': (x2-x1) * (y2-y1)
                        }
                        analysis['detection_details'].append(detail)
            
            analysis['classes_found'] = list(analysis['classes_found'])
            return analysis
            
        except Exception as e:
            print(f"Analysis failed: {e}")
            return {'error': str(e)}
        
# 4. POST-ANALYSIS METRICS - New MetricsCalculator class

class MetricsCalculator:
    """Calculate comprehensive tracking and re-identification metrics"""
    
    def __init__(self):
        self.ground_truth = {}
        self.tracking_history = []
        self.re_id_events = []
        self.id_switches = {}
        # NEW: Track re-identification events
        self.global_id_appearances = {}  # global_id -> {camera -> [frame_indices]}
        self.re_id_attempts = 0
        self.successful_re_ids = 0
        
    def log_frame_results(self, frame_idx: int, players_cam1: List[Player], 
                         players_cam2: List[Player], mappings: Dict[int, int]):
        """Enhanced logging with re-ID tracking"""
        frame_data = {
            'frame': frame_idx,
            'cam1_players': {p.id: {'global_id': p.global_id, 'bbox': p.bbox, 'confidence': p.confidence} for p in players_cam1},
            'cam2_players': {p.id: {'global_id': p.global_id, 'bbox': p.bbox, 'confidence': p.confidence} for p in players_cam2},
            'mappings': mappings,
            'timestamp': frame_idx
        }
        self.tracking_history.append(frame_data)
        
        # Track global ID appearances for re-ID analysis
        self._track_global_id_appearances(frame_idx, players_cam1, players_cam2)
    
    def _track_global_id_appearances(self, frame_idx: int, players_cam1: List[Player], players_cam2: List[Player]):
        """Track when global IDs appear in each camera"""
        for camera_id, players in [(1, players_cam1), (2, players_cam2)]:
            for player in players:
                if player.global_id is not None:
                    if player.global_id not in self.global_id_appearances:
                        self.global_id_appearances[player.global_id] = {1: [], 2: []}
                    self.global_id_appearances[player.global_id][camera_id].append(frame_idx)
    
    def calculate_id_consistency_metrics(self) -> Dict:
        """FIXED: Calculate ID consistency and re-identification metrics"""
        if len(self.tracking_history) < 2:
            return {}
        
        metrics = {
            'total_frames': len(self.tracking_history),
            'id_switches_cam1': 0,
            'id_switches_cam2': 0,
            'avg_tracking_length_cam1': 0,
            'avg_tracking_length_cam2': 0,
            're_id_success_rate': 0,
            'cross_camera_consistency': 0,
            'total_re_id_events': 0,
            'successful_re_id_events': 0
        }
        
        # Calculate re-identification success rate
        re_id_stats = self._calculate_re_id_success()
        metrics.update(re_id_stats)
        
        # Track ID consistency for each camera
        for camera in ['cam1', 'cam2']:
            player_key = f'{camera}_players'
            id_tracks = {}
            id_switches = 0
            
            prev_frame_ids = set()
            for frame_data in self.tracking_history:
                current_frame_ids = set()
                for local_id, player_info in frame_data[player_key].items():
                    global_id = player_info['global_id']
                    if global_id:
                        if global_id not in id_tracks:
                            id_tracks[global_id] = []
                        id_tracks[global_id].append(frame_data['frame'])
                        current_frame_ids.add(global_id)
                
                # Detect ID switches (simplified)
                if prev_frame_ids and current_frame_ids:
                    disappeared = prev_frame_ids - current_frame_ids
                    appeared = current_frame_ids - prev_frame_ids
                    if disappeared and appeared:
                        id_switches += min(len(disappeared), len(appeared))
                
                prev_frame_ids = current_frame_ids
            
            metrics[f'id_switches_{camera}'] = id_switches
            
            if id_tracks:
                track_lengths = [len(frames) for frames in id_tracks.values()]
                metrics[f'avg_tracking_length_{camera}'] = np.mean(track_lengths)
                metrics[f'max_tracking_length_{camera}'] = np.max(track_lengths)
                metrics[f'total_unique_players_{camera}'] = len(id_tracks)
        
        # Calculate cross-camera consistency
        cross_camera_matches = 0
        total_possible_matches = 0
        
        for frame_data in self.tracking_history:
            cam1_global_ids = set(p['global_id'] for p in frame_data['cam1_players'].values() if p['global_id'])
            cam2_global_ids = set(p['global_id'] for p in frame_data['cam2_players'].values() if p['global_id'])
            
            common_ids = cam1_global_ids.intersection(cam2_global_ids)
            cross_camera_matches += len(common_ids)
            total_possible_matches += max(len(cam1_global_ids), len(cam2_global_ids))
        
        if total_possible_matches > 0:
            metrics['cross_camera_consistency'] = cross_camera_matches / total_possible_matches
        
        return metrics
    
    def _calculate_re_id_success(self) -> Dict:
        """NEW: Calculate re-identification success metrics"""
        re_id_events = 0
        successful_re_ids = 0
        
        # Analyze global ID patterns to detect re-identification events
        for global_id, appearances in self.global_id_appearances.items():
            cam1_frames = appearances[1]
            cam2_frames = appearances[2]
            
            # Check for re-identification patterns
            if cam1_frames and cam2_frames:
                # Sort frames
                cam1_frames.sort()
                cam2_frames.sort()
                
                # Look for gaps and re-appearances (simplified re-ID detection)
                for camera_id, frames in [(1, cam1_frames), (2, cam2_frames)]:
                    if len(frames) > 1:
                        gaps = []
                        for i in range(1, len(frames)):
                            gap = frames[i] - frames[i-1]
                            if gap > 30:  # Gap of 30+ frames indicates disappearance/reappearance
                                gaps.append(gap)
                        
                        if gaps:  # Found re-appearance after disappearance
                            re_id_events += len(gaps)
                            # Check if other camera had activity during gap (successful re-ID)
                            other_camera = 2 if camera_id == 1 else 1
                            other_frames = appearances[other_camera]
                            
                            for gap_start_idx in range(len(frames) - 1):
                                gap_start = frames[gap_start_idx]
                                gap_end = frames[gap_start_idx + 1]
                                
                                # Check if other camera had this global_id during the gap
                                other_during_gap = [f for f in other_frames if gap_start < f < gap_end]
                                if other_during_gap:
                                    successful_re_ids += 1
        
        re_id_success_rate = successful_re_ids / re_id_events if re_id_events > 0 else 0
        
        return {
            'total_re_id_events': re_id_events,
            'successful_re_id_events': successful_re_ids,
            're_id_success_rate': re_id_success_rate
        }
    
    def generate_tracking_report(self) -> str:
        """Generate comprehensive tracking performance report"""
        metrics = self.calculate_id_consistency_metrics()
        
        report = f"""
PLAYER TRACKING PERFORMANCE REPORT
==================================

Basic Statistics:
- Total frames processed: {metrics.get('total_frames', 0)}
- Unique players detected (Cam1): {metrics.get('total_unique_players_cam1', 0)}
- Unique players detected (Cam2): {metrics.get('total_unique_players_cam2', 0)}

Tracking Quality:
- Average tracking length (Cam1): {metrics.get('avg_tracking_length_cam1', 0):.1f} frames
- Average tracking length (Cam2): {metrics.get('avg_tracking_length_cam2', 0):.1f} frames
- Maximum tracking length (Cam1): {metrics.get('max_tracking_length_cam1', 0)} frames
- Maximum tracking length (Cam2): {metrics.get('max_tracking_length_cam2', 0)} frames

Cross-Camera Performance:
- Cross-camera consistency: {metrics.get('cross_camera_consistency', 0):.2%}
- Re-identification success rate: {metrics.get('re_id_success_rate', 0):.2%}

ID Stability:
- ID switches (Cam1): {metrics.get('id_switches_cam1', 0)}
- ID switches (Cam2): {metrics.get('id_switches_cam2', 0)}
"""
        return report

def main():
    """Enhanced main function with better debugging"""
    # Initialize the enhanced mapper
    mapper = EnhancedCrossCameraPlayerMapper("best.pt")
    metrics_calculator = MetricsCalculator()
    
    
    # Video file paths
    video1_path = "broadcast.mp4"
    video2_path = "tacticam.mp4"
    
    # Open video captures
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open video files")
        return
    
    # Get video properties
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video 1: {fps1:.1f} FPS, {total_frames1} frames")
    print(f"Video 2: {fps2:.1f} FPS, {total_frames2} frames")
    
    frame_count = 0
    results_log = []
    detection_stats = {'cam1': [], 'cam2': []}
    
    process_interval = max(1, min(10, total_frames1 // 50))  # Process at least 50 frames
    print(f"Processing every {process_interval} frames")

    frame_count = 0  # Initialize frame counter

    try:
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break

            if frame_count % process_interval == 0:
                try:
                    # Process frames with enhanced tracking
                    result = mapper.process_frames(frame1, frame2)
                    if result is None:
                        print(f"Warning: process_frames returned None at frame {frame_count}")
                        continue
                    players_cam1, players_cam2, matches = result

                    # Log for metrics calculation
                    metrics_calculator.log_frame_results(frame_count, players_cam1, players_cam2, matches)

                    # Analyze detection quality for debugging (every 10th processed frame)
                    if frame_count % (process_interval * 10) == 0:
                        analysis1 = mapper.analyze_detection_quality(frame1)
                        analysis2 = mapper.analyze_detection_quality(frame2)
                        print(f"Frame {frame_count} analysis:")
                        print(f"  Cam1: {analysis1['total_detections']} total, {analysis1['player_detections']} players, {analysis1['filtered_players']} filtered")
                        print(f"  Cam2: {analysis2['total_detections']} total, {analysis2['player_detections']} players, {analysis2['filtered_players']} filtered")

                    # Log results
                    result_entry = {
                        'frame': frame_count,
                        'cam1_players': len(players_cam1),
                        'cam2_players': len(players_cam2),
                        'matches': len(matches),
                        'mappings': matches
                    }
                    results_log.append(result_entry)

                    detection_stats['cam1'].append(len(players_cam1))
                    detection_stats['cam2'].append(len(players_cam2))

                    print(f"Frame {frame_count}: Cam1={len(players_cam1)}, Cam2={len(players_cam2)}, Matches={len(matches)}")
                    if matches:
                        print(f"  Matches: {matches}")

                    # Draw detections
                    output1 = mapper.draw_detections(frame1, players_cam1, matches, True)
                    output2 = mapper.draw_detections(frame2, players_cam2, matches, False)

                    # Resize and combine frames for display
                    height = 480
                    width1 = int(output1.shape[1] * height / output1.shape[0])
                    width2 = int(output2.shape[1] * height / output2.shape[0])

                    output1_resized = cv2.resize(output1, (width1, height))
                    output2_resized = cv2.resize(output2, (width2, height))

                    combined = np.hstack([output1_resized, output2_resized])

                    # Add titles and stats
                    cv2.putText(combined, "Broadcast View", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(combined, "Tactical View", (width1 + 10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    stats_text = f"Frame: {frame_count} | Players: {len(players_cam1)}+{len(players_cam2)} | Matches: {len(matches)}"
                    cv2.putText(combined, stats_text, (10, combined.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Display
                    cv2.imshow('Enhanced Cross-Camera Player Mapping', combined)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Stopping processing...")
                        break
                    elif key == ord('p'):  # Pause
                        cv2.waitKey(0)

                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
                    traceback.print_exc()
                    continue

            frame_count += 1

    except KeyboardInterrupt:
        print("Processing interrupted by user")

    finally:
        # Generate comprehensive report
        tracking_report = metrics_calculator.generate_tracking_report()
        print(tracking_report)

        # Save detailed metrics
        with open('tracking_metrics_report.txt', 'w') as f:
            f.write(tracking_report)

        # Cleanup
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

        # Save detailed results
        summary = {
            'total_frames_processed': len(results_log),
            'average_detections_cam1': np.mean(detection_stats['cam1']) if detection_stats['cam1'] else 0,
            'average_detections_cam2': np.mean(detection_stats['cam2']) if detection_stats['cam2'] else 0,
            'total_matches': sum(r['matches'] for r in results_log),
            'results': results_log
        }

        with open('enhanced_player_mapping_results.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nProcessing complete!")
        print(f"Processed {len(results_log)} frames")
        print(f"Average detections - Cam1: {summary['average_detections_cam1']:.1f}, Cam2: {summary['average_detections_cam2']:.1f}")
        print(f"Total matches found: {summary['total_matches']}")
        print("Detailed results saved to enhanced_player_mapping_results.json")


if __name__ == "__main__":
    main()