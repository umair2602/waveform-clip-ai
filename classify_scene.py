#!/usr/bin/env python3
"""
Command-line interface for scene classification.

This script provides an easy-to-use CLI for classifying images into 6 scene types.
Supports single images, batch processing, and multiple output formats.
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from typing import List
from scene_classifier import SceneClassifier


def find_images(directory: Path, recursive: bool = False) -> List[Path]:
    """Find all image files in a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    if recursive:
        images = [
            f for f in directory.rglob('*')
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
    else:
        images = [
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
    
    return sorted(images)


def format_text_output(results: dict, classifier: SceneClassifier, verbose: bool = False) -> str:
    """Format results as human-readable text."""
    output = []
    
    for image_path, predictions in results.items():
        output.append(f"\n{'=' * 80}")
        output.append(f"Image: {image_path}")
        output.append('-' * 80)
        
        if "error" in predictions:
            output.append(f"ERROR: {predictions['error']}")
            continue
        
        # Top prediction
        top_type = list(predictions.keys())[0]
        top_conf = predictions[top_type]
        output.append(f"\n🎯 Top Prediction:")
        output.append(f"   {classifier.get_scene_description(top_type)}")
        output.append(f"   Confidence: {top_conf:.2f}%")
        
        # All predictions if verbose
        if verbose and len(predictions) > 1:
            output.append(f"\n📊 All Predictions:")
            for scene_type, confidence in predictions.items():
                description = classifier.get_scene_description(scene_type)
                bar_length = int(confidence / 2)  # Scale to 50 chars max
                bar = '█' * bar_length
                output.append(f"   {confidence:5.2f}% {bar:50s} {scene_type}")
    
    output.append('=' * 80)
    return '\n'.join(output)


def format_json_output(results: dict, classifier: SceneClassifier) -> str:
    """Format results as JSON."""
    # Add descriptions to output
    output = {}
    for image_path, predictions in results.items():
        if "error" in predictions:
            output[image_path] = predictions
        else:
            output[image_path] = {
                "predictions": predictions,
                "top_prediction": {
                    "type": list(predictions.keys())[0],
                    "confidence": list(predictions.values())[0],
                    "description": classifier.get_scene_description(list(predictions.keys())[0])
                }
            }
    
    return json.dumps(output, indent=2)


def format_csv_output(results: dict, classifier: SceneClassifier) -> str:
    """Format results as CSV."""
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    header = ['image_path', 'top_scene_type', 'confidence', 'description']
    all_scene_types = SceneClassifier.get_available_scene_types()
    header.extend([f'{st}_confidence' for st in all_scene_types])
    writer.writerow(header)
    
    # Data rows
    for image_path, predictions in results.items():
        if "error" in predictions:
            writer.writerow([image_path, 'ERROR', 0, predictions['error']])
            continue
        
        top_type = list(predictions.keys())[0]
        top_conf = predictions[top_type]
        description = classifier.get_scene_description(top_type)
        
        row = [image_path, top_type, f"{top_conf:.2f}", description]
        
        # Add confidence for each scene type
        for st in all_scene_types:
            conf = predictions.get(st, 0.0)
            row.append(f"{conf:.2f}")
        
        writer.writerow(row)
    
    return output.getvalue()


def main():
    parser = argparse.ArgumentParser(
        description='Classify images into 6 scene types using CLIP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scene Types:
  indoor_small  - Small enclosed spaces (bedroom, office, closet)
  indoor_large  - Large spaces with many objects (warehouse, workshop)
  outdoor       - Exterior scenes (street, park, parking lot)
  close_up      - Macro shots showing fine detail (<1m)
  far_away      - Distant objects (>15m away)
  glass_shiny   - Scenes with reflective surfaces (windows, mirrors)

Examples:
  # Classify single image
  python classify_scene.py --image photo.jpg
  
  # Classify all images in directory with verbose output
  python classify_scene.py --directory images/ --verbose
  
  # Export results to JSON
  python classify_scene.py --directory images/ --format json --output results.json
  
  # Show only predictions above 10% confidence
  python classify_scene.py --image photo.jpg --threshold 10 --verbose
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--image', '-i',
        type=Path,
        help='Path to a single image file'
    )
    input_group.add_argument(
        '--directory', '-d',
        type=Path,
        help='Path to directory containing images'
    )
    input_group.add_argument(
        '--list-types', '-l',
        action='store_true',
        help='List all available scene types and exit'
    )
    
    # Processing options
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Process directories recursively (only with --directory)'
    )
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=None,
        help='Show only top K predictions (default: all)'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.0,
        help='Minimum confidence threshold (0-100, default: 0)'
    )
    
    # Output options
    parser.add_argument(
        '--format', '-f',
        choices=['text', 'json', 'csv'],
        default='text',
        help='Output format (default: text)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output file path (default: stdout)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed predictions for all scene types'
    )
    
    # Model options
    parser.add_argument(
        '--model',
        default='ViT-B/32',
        choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
        help='CLIP model variant (default: ViT-B/32)'
    )
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to run on (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Handle --list-types
    if args.list_types:
        SceneClassifier.print_scene_types()
        return 0
    
    # Initialize classifier
    device = None if args.device == 'auto' else args.device
    classifier = SceneClassifier(model_name=args.model, device=device)
    
    # Collect images to process
    if args.image:
        if not args.image.exists():
            print(f"Error: Image file not found: {args.image}", file=sys.stderr)
            return 1
        image_paths = [args.image]
    else:  # args.directory
        if not args.directory.exists():
            print(f"Error: Directory not found: {args.directory}", file=sys.stderr)
            return 1
        if not args.directory.is_dir():
            print(f"Error: Not a directory: {args.directory}", file=sys.stderr)
            return 1
        
        image_paths = find_images(args.directory, args.recursive)
        if not image_paths:
            print(f"Error: No images found in {args.directory}", file=sys.stderr)
            return 1
        
        print(f"Found {len(image_paths)} images to process...")
    
    # Classify images
    print("Classifying images...")
    results = classifier.classify_batch(
        image_paths,
        top_k=args.top_k,
        threshold=args.threshold
    )
    
    # Format output
    if args.format == 'text':
        output_str = format_text_output(results, classifier, args.verbose)
    elif args.format == 'json':
        output_str = format_json_output(results, classifier)
    else:  # csv
        output_str = format_csv_output(results, classifier)
    
    # Write output
    if args.output:
        args.output.write_text(output_str)
        print(f"\nResults written to: {args.output}")
    else:
        print(output_str)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
