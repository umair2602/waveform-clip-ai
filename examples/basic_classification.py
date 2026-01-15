"""
Example: Basic single image classification

This example shows how to classify a single image and get the top prediction.
"""

from scene_classifier import SceneClassifier
from pathlib import Path

def main():
    # Initialize the classifier
    print("Initializing CLIP scene classifier...")
    classifier = SceneClassifier()
    
    # Specify your image path
    image_path = "test_image.jpg"  # Replace with your image path
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image not found: {image_path}")
        print("\nPlease update the image_path variable in this script.")
        return
    
    # Classify the image
    print(f"\nClassifying: {image_path}")
    print("-" * 60)
    
    # Get all predictions
    results = classifier.classify_image(image_path)
    
    print("\n📊 All Scene Type Predictions:")
    for scene_type, confidence in results.items():
        description = classifier.get_scene_description(scene_type)
        print(f"  {confidence:5.2f}% - {description}")
    
    # Get top prediction
    print("\n🎯 Top Prediction:")
    top_type, top_confidence = classifier.get_top_prediction(image_path)
    print(f"  Scene Type: {top_type}")
    print(f"  Description: {classifier.get_scene_description(top_type)}")
    print(f"  Confidence: {top_confidence:.2f}%")
    
    # Interpretation guidance
    print("\n💡 Interpretation:")
    if top_confidence > 50:
        print("  High confidence - the classifier is quite sure about this classification")
    elif top_confidence > 30:
        print("  Moderate confidence - reasonably confident prediction")
    else:
        print("  Low confidence - the image may match multiple scene types")

if __name__ == "__main__":
    main()
