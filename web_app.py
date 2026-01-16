"""
Web interface for CLIP scene classification.

A Flask-based web application that allows users to upload images
and get real-time scene classification results.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import os
import base64
from io import BytesIO
from PIL import Image
import cv2
from scene_classifier import SceneClassifier

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = Path('uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm', 'm4v'}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max for batch uploads

# Initialize CLIP classifier (load once at startup)
print("Initializing CLIP scene classifier...")
classifier = SceneClassifier()
print("Classifier ready!")


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image_path):
    """Convert image to base64 for display in browser."""
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def is_video_file(filename):
    """Check if file is a video."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def extract_frames_from_video(video_path, num_frames=10):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract (evenly spaced)
        
    Returns:
        List of (frame_path, frame_number) tuples
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    # Calculate frame indices to extract (evenly spaced)
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    extracted_frames = []
    video_name = Path(video_path).stem
    
    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Save frame as temporary image
            frame_path = UPLOAD_FOLDER / f"{video_name}_frame_{idx:03d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            extracted_frames.append((str(frame_path), frame_idx))
    
    cap.release()
    
    return extracted_frames, {
        'total_frames': total_frames,
        'fps': fps,
        'duration': duration,
        'extracted_count': len(extracted_frames)
    }


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/api/classify', methods=['POST'])
def classify():
    """API endpoint to classify uploaded image."""
    # Check if file was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    # Check if file was selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Classify the image
        results = classifier.classify_image(filepath)
        
        # Get top prediction
        top_type, top_confidence = classifier.get_top_prediction(filepath)
        
        # Get model recommendation
        recommended_model = classifier.get_model_recommendation(top_type, prefer_speed="balanced")
        
        # Prepare response
        predictions = []
        for scene_type, confidence in results.items():
            predictions.append({
                'type': scene_type,
                'description': classifier.get_scene_description(scene_type),
                'confidence': round(confidence, 2)
            })
        
        # Convert image to base64 for display
        image_data = image_to_base64(filepath)
        
        response = {
            'success': True,
            'image': f'data:image/jpeg;base64,{image_data}',
            'top_prediction': {
                'type': top_type,
                'description': classifier.get_scene_description(top_type),
                'confidence': round(top_confidence, 2),
                'recommended_model': recommended_model  # NEW: Include model recommendation
            },
            'all_predictions': predictions
        }
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': f'Classification failed: {str(e)}'}), 500


@app.route('/api/classify-batch', methods=['POST'])
def classify_batch():
    """API endpoint to classify multiple uploaded images."""
    # Check if files were uploaded
    if 'images' not in request.files:
        return jsonify({'error': 'No image files provided'}), 400
    
    images = request.files.getlist('images')
    if not images:
        return jsonify({'error': 'No images in request'}), 400
    
    try:
        results = []
        scene_counts = {}
        
        for img_file in images:
            # Skip invalid files
            if not img_file or img_file.filename == '' or not allowed_file(img_file.filename):
                continue
            
            # Save and classify each image
            filename = secure_filename(img_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img_file.save(filepath)
            
            # Classify
            classifications = classifier.classify_image(filepath)
            top_type, top_conf = classifier.get_top_prediction(filepath)
            
            # Count scene types
            scene_counts[top_type] = scene_counts.get(top_type, 0) + 1
            
            # Convert image to base64 for display
            image_data = image_to_base64(filepath)
            
            results.append({
                'filename': filename,
                'image': f'data:image/jpeg;base64,{image_data}',
                'scene_type': top_type,
                'confidence': round(top_conf, 2),
                'all_predictions': {
                    scene: round(conf, 2) 
                    for scene, conf in classifications.items()
                }
            })
            
            # Clean up
            os.remove(filepath)
        
        if not results:
            return jsonify({'error': 'No valid images provided'}), 400
        
        # Calculate dataset summary
        total_images = len(results)
        scene_distribution = {
            scene: (count / total_images) * 100 
            for scene, count in scene_counts.items()
        }
        
        # Get dominant scene types (>20% of dataset)
        dominant_scenes = [
            scene for scene, pct in scene_distribution.items() 
            if pct > 20
        ]
        
        # If no dominant scenes, use top 2
        if not dominant_scenes:
            dominant_scenes = sorted(
                scene_distribution.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:2]
            dominant_scenes = [scene for scene, _ in dominant_scenes]
        
        # Recommend models for dominant scenes
        recommended_models = []
        for scene in dominant_scenes:
            model = classifier.get_model_recommendation(scene)
            recommended_models.append({
                'scene_type': scene,
                'model': model,
                'percentage': round(scene_distribution[scene], 1)
            })
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': {
                'total_images': total_images,
                'scene_distribution': scene_distribution,
                'dominant_scenes': dominant_scenes,
                'recommended_models': recommended_models
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Batch classification failed: {str(e)}'}), 500


@app.route('/api/classify-video', methods=['POST'])
def classify_video():
    """API endpoint to classify frames from uploaded video."""
    # Check if file was uploaded
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    # Check if file was selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file type is allowed
    if not is_video_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: ' + ', '.join(ALLOWED_VIDEO_EXTENSIONS)}), 400
    
    try:
        # Get number of frames from request (default: 10)
        num_frames = int(request.form.get('num_frames', 10))
        num_frames = min(max(num_frames, 1), 50)  # Limit between 1-50
        
        # Save uploaded video
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        
        # Extract frames
        frames, video_info = extract_frames_from_video(video_path, num_frames)
        
        if not frames:
            os.remove(video_path)
            return jsonify({'error': 'Could not extract frames from video'}), 400
        
        # Classify each frame
        results = []
        scene_counts = {}
        
        for frame_path, frame_idx in frames:
            # Classify
            classifications = classifier.classify_image(frame_path)
            top_type, top_conf = classifier.get_top_prediction(frame_path)
            
            # Count scene types
            scene_counts[top_type] = scene_counts.get(top_type, 0) + 1
            
            # Convert frame to base64 for display
            image_data = image_to_base64(frame_path)
            
            results.append({
                'frame_index': frame_idx,
                'timestamp': frame_idx / video_info['fps'] if video_info['fps'] > 0 else 0,
                'image': f'data:image/jpeg;base64,{image_data}',
                'scene_type': top_type,
                'confidence': round(top_conf, 2),
                'all_predictions': {
                    scene: round(conf, 2) 
                    for scene, conf in classifications.items()
                }
            })
            
            # Clean up frame
            os.remove(frame_path)
        
        # Clean up video
        os.remove(video_path)
        
        # Calculate summary
        total_frames = len(results)
        scene_distribution = {
            scene: (count / total_frames) * 100 
            for scene, count in scene_counts.items()
        }
        
        # Get dominant scenes
        dominant_scenes = [
            scene for scene, pct in scene_distribution.items() 
            if pct > 20
        ]
        
        if not dominant_scenes:
            dominant_scenes = sorted(
                scene_distribution.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:2]
            dominant_scenes = [scene for scene, _ in dominant_scenes]
        
        # Recommend models for dominant scenes
        recommended_models = []
        for scene in dominant_scenes:
            model = classifier.get_model_recommendation(scene)
            recommended_models.append({
                'scene_type': scene,
                'model': model,
                'percentage': round(scene_distribution[scene], 1)
            })
        
        return jsonify({
            'success': True,
            'video_info': video_info,
            'results': results,
            'summary': {
                'total_frames': total_frames,
                'scene_distribution': scene_distribution,
                'dominant_scenes': dominant_scenes,
                'recommended_models': recommended_models
            }
        })
    
    except Exception as e:
        return jsonify({'error': f'Video classification failed: {str(e)}'}), 500


@app.route('/api/scene-types', methods=['GET'])
def get_scene_types():
    """Get all available scene types."""
    scene_types = []
    for scene_type in SceneClassifier.get_available_scene_types():
        scene_types.append({
            'type': scene_type,
            'description': classifier.get_scene_description(scene_type)
        })
    return jsonify({'scene_types': scene_types})


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 CLIP Scene Classifier Web Interface")
    print("="*60)
    print("\nStarting server...")
    print("Open your browser and navigate to: http://localhost:8080")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=8080)
