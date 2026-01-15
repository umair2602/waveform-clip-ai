"""
Scene Classifier using CLIP

This module provides scene classification for images into 6 specific categories:
1. Indoor Small - Small enclosed spaces
2. Indoor Large - Large indoor spaces with high object density
3. Outdoor - Exterior scenes
4. Close-up - Macro/near-field shots (<1m)
5. Far-away - Distant subjects (>15m)
6. Glass/Shiny - Scenes with reflective surfaces
"""

import torch
import clip
from PIL import Image
from typing import List, Dict, Tuple, Union
from pathlib import Path
import numpy as np


class SceneClassifier:
    """CLIP-based scene classifier for 6 scene types."""
    
    # Define scene types with multiple text prompt variations for better accuracy
    SCENE_TYPES = {
        "indoor_small": [
            "a photo of a small indoor room",
            "a bedroom interior",
            "an office cubicle",
            "a small closet",
            "a compact indoor space with limited area",
            "a confined indoor environment",
        ],
        "indoor_large": [
            "a photo of a large indoor space with many objects",
            "a warehouse interior full of items",
            "a cluttered living room with lots of furniture",
            "a workshop filled with tools and equipment",
            "a spacious indoor area with high object density",
            "a crowded indoor environment with numerous items",
        ],
        "outdoor": [
            "a photo of an outdoor scene",
            "a street view",
            "a park landscape",
            "a parking lot",
            "an exterior environment with natural lighting",
            "an open air scene",
        ],
        "close_up": [
            "a photo of a close-up object",
            "a macro shot showing fine detail",
            "a water bottle photographed up close",
            "a near-field shot with visible texture",
            "an extreme close-up showing surface details",
            "a detailed view of a small object",
        ],
        "far_away": [
            "a photo of distant objects more than 15 meters away",
            "a building across the street",
            "a mountain in the distance",
            "a car far away",
            "a long-range view of distant scenery",
            "objects far in the background",
        ],
        "glass_shiny": [
            "a photo with glass or shiny reflective surfaces",
            "a scene with windows and reflections",
            "mirrors and reflective materials",
            "polished metal surfaces",
            "a scene with glare and distortion from glass",
            "reflective surfaces with low texture",
        ],
    }
    
    SCENE_DESCRIPTIONS = {
        "indoor_small": "Indoor scene (small room) - e.g., bedroom, office cubicle, closet",
        "indoor_large": "Indoor scene (large room with lots of objects) - e.g., warehouse, cluttered living room, workshop",
        "outdoor": "Outdoor scene - e.g., street, park, parking lot",
        "close_up": "Close-up scene - macro or near-field shots (<1m)",
        "far_away": "Far-away scene - distant objects (>15m)",
        "glass_shiny": "Scene with glass or shiny surfaces - reflective materials",
    }
    
    # Model recommendations for each scene type
    MODEL_RECOMMENDATIONS = {
        "indoor_small": {
            "primary": {
                "name": "LightGlue",
                "speed": "fast",
                "reason": "Excellent speed/accuracy for indoor scenes with abundant close-range features"
            },
            "alternative": {
                "name": "SuperGlue",
                "speed": "medium",
                "reason": "Better for low-texture indoor areas with complex geometry"
            },
            "fast_option": {
                "name": "SIFT",
                "speed": "fast",
                "reason": "Reliable traditional fallback for simple indoor matching"
            }
        },
        "indoor_large": {
            "primary": {
                "name": "SuperGlue",
                "speed": "medium",
                "reason": "Robust to clutter and occlusions in complex environments"
            },
            "alternative": {
                "name": "OmniGlue",
                "speed": "medium",
                "reason": "Handles complex occlusions and varied depths well"
            },
            "dense_option": {
                "name": "DUSt3R",
                "speed": "medium",
                "reason": "Dense reconstruction for complete scene coverage"
            }
        },
        "outdoor": {
            "primary": {
                "name": "XFeat",
                "speed": "fast",
                "reason": "Scale-invariant and robust to lighting changes"
            },
            "alternative": {
                "name": "xfeat+lightglue",
                "speed": "fast",
                "reason": "Enhanced accuracy for outdoor scenes with wide baselines"
            },
            "dense_option": {
                "name": "DUSt3R",
                "speed": "medium",
                "reason": "Dense matching for large outdoor scenes"
            }
        },
        "close_up": {
            "primary": {
                "name": "DISK",
                "speed": "slow",
                "reason": "Dense features capture high texture detail in macro shots"
            },
            "alternative": {
                "name": "LightGlue",
                "speed": "fast",
                "reason": "Fast matching for texture-rich close-up scenes"
            },
            "detail_option": {
                "name": "DeDoDe",
                "speed": "medium",
                "reason": "Captures fine details in close-up imagery"
            }
        },
        "far_away": {
            "primary": {
                "name": "SIFT",
                "speed": "fast",
                "reason": "Robust scale invariance for distant, low-texture objects"
            },
            "alternative": {
                "name": "XoFTR",
                "speed": "medium",
                "reason": "Handles low-texture distant features well"
            },
            "sparse_option": {
                "name": "RDD(sparse)",
                "speed": "medium",
                "reason": "Efficient sparse matching for distant objects"
            }
        },
        "glass_shiny": {
            "primary": {
                "name": "XoFTR",
                "speed": "medium",
                "reason": "Handles geometric transforms from reflections and distortions"
            },
            "alternative": {
                "name": "RIPE",
                "speed": "medium",
                "reason": "Robust to specular reflections and glare"
            },
            "specialized": {
                "name": "Mast3R",
                "speed": "slow",
                "reason": "3D-aware matching handles complex reflection distortions"
            }
        }
    }
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Initialize the scene classifier.
        
        Args:
            model_name: CLIP model variant to use (default: ViT-B/32)
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading CLIP model '{model_name}' on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        # Prepare all text prompts
        self._prepare_text_features()
        
    def _prepare_text_features(self):
        """Pre-compute text features for all scene type prompts."""
        self.text_features_by_type = {}
        
        with torch.no_grad():
            for scene_type, prompts in self.SCENE_TYPES.items():
                # Tokenize all prompts for this scene type
                text_tokens = clip.tokenize(prompts).to(self.device)
                
                # Encode text features
                features = self.model.encode_text(text_tokens)
                features /= features.norm(dim=-1, keepdim=True)
                
                # Average features across all prompts for this scene type
                avg_features = features.mean(dim=0, keepdim=True)
                avg_features /= avg_features.norm(dim=-1, keepdim=True)
                
                self.text_features_by_type[scene_type] = avg_features
                
    def classify_image(
        self, 
        image_path: Union[str, Path], 
        top_k: int = None,
        threshold: float = 0.0
    ) -> Dict[str, float]:
        """
        Classify a single image into scene types.
        
        Args:
            image_path: Path to the image file
            top_k: Return only top K predictions (None = all)
            threshold: Minimum confidence threshold (0.0-1.0)
            
        Returns:
            Dictionary mapping scene types to confidence scores (0-100)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Encode image
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarities for each scene type
        similarities = {}
        for scene_type, text_features in self.text_features_by_type.items():
            similarity = (100.0 * image_features @ text_features.T).squeeze()
            similarities[scene_type] = similarity.cpu().item()
        
        # Apply softmax to get probabilities
        scores_tensor = torch.tensor(list(similarities.values()))
        probabilities = torch.softmax(scores_tensor, dim=0).numpy()
        
        # Create results dictionary with probabilities
        results = {
            scene_type: float(prob * 100)  # Convert to percentage
            for scene_type, prob in zip(similarities.keys(), probabilities)
        }
        
        # Filter by threshold
        if threshold > 0:
            results = {k: v for k, v in results.items() if v >= threshold}
        
        # Sort by confidence
        results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
        # Return top K if specified
        if top_k is not None:
            results = dict(list(results.items())[:top_k])
        
        return results
    
    def classify_batch(
        self,
        image_paths: List[Union[str, Path]],
        top_k: int = None,
        threshold: float = 0.0
    ) -> Dict[str, Dict[str, float]]:
        """
        Classify multiple images.
        
        Args:
            image_paths: List of image file paths
            top_k: Return only top K predictions per image
            threshold: Minimum confidence threshold
            
        Returns:
            Dictionary mapping image paths to classification results
        """
        results = {}
        for image_path in image_paths:
            try:
                path_str = str(image_path)
                results[path_str] = self.classify_image(image_path, top_k, threshold)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results[str(image_path)] = {"error": str(e)}
        
        return results
    
    def get_top_prediction(self, image_path: Union[str, Path]) -> Tuple[str, float]:
        """
        Get the single most likely scene type for an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (scene_type, confidence_percentage)
        """
        results = self.classify_image(image_path, top_k=1)
        scene_type = list(results.keys())[0]
        confidence = results[scene_type]
        return scene_type, confidence
    
    def get_scene_description(self, scene_type: str) -> str:
        """Get human-readable description for a scene type."""
        return self.SCENE_DESCRIPTIONS.get(scene_type, scene_type)
    
    def get_model_recommendation(self, scene_type: str, prefer_speed: str = "balanced") -> Dict:
        """
        Get recommended image matching model based on scene type.
        
        Args:
            scene_type: The classified scene type
            prefer_speed: Preference for speed ("fast", "balanced", "accurate")
            
        Returns:
            Dictionary with model name, speed rating, and reasoning
        """
        recommendations = self.MODEL_RECOMMENDATIONS.get(scene_type, {})
        
        if not recommendations:
            # Fallback to SIFT if scene type not found
            return {
                "name": "SIFT",
                "speed": "fast",
                "reason": "Universal fallback - works for most scenes"
            }
        
        # Select based on speed preference
        if prefer_speed == "fast":
            # Prefer fast_option if available, otherwise primary
            model = recommendations.get("fast_option") or recommendations.get("primary")
        elif prefer_speed == "accurate":
            # Prefer dense/detail/specialized options if available
            model = (recommendations.get("dense_option") or 
                    recommendations.get("detail_option") or
                    recommendations.get("specialized") or
                    recommendations.get("alternative"))
        else:  # balanced
            # Use primary recommendation
            model = recommendations.get("primary")
        
        return model if model else recommendations.get("primary")
    
    @staticmethod
    def get_available_scene_types() -> List[str]:
        """Get list of all available scene types."""
        return list(SceneClassifier.SCENE_TYPES.keys())
    
    @staticmethod
    def print_scene_types():
        """Print all available scene types with descriptions."""
        print("\nAvailable Scene Types:")
        print("=" * 80)
        for scene_type, description in SceneClassifier.SCENE_DESCRIPTIONS.items():
            print(f"\n{scene_type}:")
            print(f"  {description}")
        print("\n" + "=" * 80)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python scene_classifier.py <image_path>")
        SceneClassifier.print_scene_types()
        sys.exit(1)
    
    # Initialize classifier
    classifier = SceneClassifier()
    
    # Classify image
    image_path = sys.argv[1]
    print(f"\nClassifying: {image_path}")
    print("-" * 80)
    
    results = classifier.classify_image(image_path)
    
    print("\nPredictions (sorted by confidence):")
    for scene_type, confidence in results.items():
        description = classifier.get_scene_description(scene_type)
        print(f"  {confidence:5.2f}% - {description}")
    
    print("\nTop Prediction:")
    top_type, top_confidence = classifier.get_top_prediction(image_path)
    print(f"  {classifier.get_scene_description(top_type)}")
    print(f"  Confidence: {top_confidence:.2f}%")
