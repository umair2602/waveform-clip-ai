# CLIP Scene Classifier with Intelligent Model Recommendations

**Automated scene classification and matching model recommendation system for 3D reconstruction pipelines.**

---

## 🎯 What It Does

This system analyzes your images to:
1. **Classify scene types** using OpenAI's CLIP model (6 categories)
2. **Recommend optimal image matching models** for 3D reconstruction
3. **Process batches** to provide dataset-level insights and aggregate recommendations

---

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- `pip` package manager

### Installation

```bash
# 1. Clone/navigate to the project
cd /path/to/clip

# 2. Install dependencies
pip install -r requirements_classifier.txt

# 3. Start the web server
python3 web_app.py

# 4. Open browser
# Navigate to: http://localhost:8080
```

---

## 📁 Project Structure

```
clip/
├── scene_classifier.py          # Core CLIP classifier with model recommendations
├── web_app.py                   # Flask web server (single + batch endpoints)
├── classify_scene.py            # CLI tool for command-line usage
│
├── templates/
│   └── index.html               # Web interface HTML
│
├── static/
│   ├── css/
│   │   └── style.css           # Modern UI styling
│   └── js/
│       └── app.js              # Frontend logic (single + batch processing)
│
├── CLIP/                        # OpenAI CLIP implementation
├── requirements_classifier.txt  # Python dependencies
└── README.md                    # This file
```

---

## 🔄 System Flow

### Architecture Overview

```
┌─────────────┐
│   User      │
│  Browser    │
└──────┬──────┘
       │
       │ HTTP (upload images)
       ▼
┌─────────────────────────────────────────┐
│           Flask Web Server              │
│         (web_app.py:8080)              │
│                                         │
│  Routes:                                │
│  • GET  /                → index.html   │
│  • POST /api/classify   → single image │
│  • POST /api/classify-batch → multiple │
└──────────┬──────────────────────────────┘
           │
           │ calls
           ▼
┌─────────────────────────────────────────┐
│       Scene Classifier                  │
│      (scene_classifier.py)             │
│                                         │
│  • Loads CLIP ViT-B/32 model           │
│  • Classifies images into 6 scenes     │
│  • Returns model recommendations        │
└──────────┬──────────────────────────────┘
           │
           │ uses
           ▼
┌─────────────────────────────────────────┐
│         OpenAI CLIP Model               │
│          (CLIP/ folder)                 │
│                                         │
│  • Vision-Language model                │
│  • Zero-shot image classification       │
│  • Text-image similarity matching       │
└─────────────────────────────────────────┘
```

### Single Image Flow

1. **User uploads image** via web UI or drag-and-drop
2. **Frontend (app.js)** sends image to `/api/classify`
3. **Backend (web_app.py)** saves image temporarily
4. **Classifier (scene_classifier.py)**:
   - Loads and preprocesses image
   - Encodes with CLIP vision encoder
   - Compares with pre-computed text embeddings (36 prompts)
   - Calculates softmax probabilities
   - Identifies top scene type
5. **Model Recommendation**:
   - Looks up scene type in `MODEL_RECOMMENDATIONS`
   - Returns optimal matching model (name, speed, reason)
6. **Response** returned as JSON with:
   - Classified scene type + confidence
   - All predictions with scores
   - Recommended matching model
7. **Frontend displays** results with visual cards

### Batch Processing Flow

1. **User selects multiple images** (2-50 recommended)
2. **Frontend (app.js)** sends all images to `/api/classify-batch`
3. **Backend (web_app.py)** processes each image sequentially
4. **Aggregation**:
   - Counts scene type occurrences
   - Calculates scene distribution percentages
   - Identifies dominant scenes (>20% of dataset)
5. **Dataset-level recommendations**:
   - For each dominant scene, suggest best model
   - Returns aggregate recommendations with percentages
6. **Response** includes:
   - Individual results for each image
   - Scene distribution chart data
   - Recommended models for dataset
7. **Frontend renders**:
   - Dataset summary statistics
   - Scene distribution bar chart
   - Model recommendation cards
   - Image grid with individual results

---

## 🏗️ File-by-File Breakdown

### `scene_classifier.py` (Core Logic)

**Purpose**: CLIP-based scene classifier with model recommendations

**Key Components**:

```python
class SceneClassifier:
    # Constants
    SCENE_TYPES = {
        "indoor_small": [...prompts...],
        "outdoor": [...prompts...],
        # 6 scene types, 6 prompts each
    }
    
    MODEL_RECOMMENDATIONS = {
        "indoor_small": {
            "primary": {"name": "LightGlue", "speed": "fast", ...},
            "alternative": {...},
            "fast_option": {...}
        },
        # Mapped for all 6 scene types
    }
    
    # Methods
    __init__()                    # Load CLIP model, prepare text embeddings
    classify_image()              # Classify single image
    classify_batch()              # Classify multiple images
    get_top_prediction()          # Get highest confidence scene
    get_model_recommendation()    # Get matching model for scene
```

**Flow**:
1. Initialization: Load CLIP ViT-B/32, encode all text prompts (36 total)
2. Classification: Encode image, compute cosine similarity, apply softmax
3. Recommendation: Lookup scene type, return optimal model config

---

### `web_app.py` (Web Server)

**Purpose**: Flask HTTP server with REST API

**Configuration**:
- Port: `8080`
- Max upload: `100MB` (for batch uploads)
- Upload folder: `./uploads` (temporary)

**API Endpoints**:

#### `GET /`
- Serves `templates/index.html`
- Main web interface

#### `POST /api/classify`
- **Input**: Single image file (multipart/form-data)
- **Output**: JSON with classification + model recommendation
- **Process**:
  1. Validate file type and size
  2. Save temporarily
  3. Call `classifier.classify_image()`
  4. Get model recommendation
  5. Convert image to base64 for display
  6. Clean up temp file
  7. Return JSON response

#### `POST /api/classify-batch`
- **Input**: Multiple image files (field name: `images`)
- **Output**: JSON with individual results + dataset summary
- **Process**:
  1. Validate all files
  2. For each image: classify and count scene types
  3. Calculate scene distribution percentages
  4. Identify dominant scenes (>20% or top 2)
  5. Get model recommendations for dominant scenes
  6. Return aggregated results

#### `GET /api/scene-types`
- Returns list of available scene types with descriptions

---

### `classify_scene.py` (CLI Tool)

**Purpose**: Command-line interface for batch processing without web server

**Usage**:
```bash
# Single image
python classify_scene.py image.jpg

# Multiple images
python classify_scene.py *.jpg

# JSON output
python classify_scene.py --output json image.jpg

# Show all predictions
python classify_scene.py --all image.jpg
```

**Features**:
- Batch processing from command line
- Multiple output formats (text, JSON, CSV)
- Verbose mode for detailed predictions
- Progress indicators

---

### `templates/index.html` (Web UI)

**Purpose**: Frontend HTML structure

**Sections**:
- Header with title
- Upload area (drag-and-drop + click)
- Loading state with spinner
- Single image results display
- Batch results container (dynamically created)
- Scene type definitions
- Error handling

---

### `static/js/app.js` (Frontend Logic)

**Purpose**: Client-side JavaScript for UI interactions

**Key Functions**:

```javascript
handleFiles(files)              // Route to single or batch upload
uploadAndClassify(file)         // Single image upload
uploadAndClassifyBatch(files)   // Batch upload
displayResults(data)            // Show single image results
displayBatchResults(data)       // Show dataset analysis
```

**Features**:
- File validation (type, size)
- Drag-and-drop support
- Loading state management
- Dynamic HTML generation
- Error handling and user feedback

---

### `static/css/style.css` (Styling)

**Purpose**: Modern, premium UI design

**Design System**:
- Dark theme (`--bg-main: #0f172a`)
- Accent colors (`--primary: #6366f1`)
- Speed badges (green/yellow/red)
- Responsive grid layouts
- Smooth animations and transitions

---

## 🎨 Scene Types & Model Recommendations

### Scene Classification

| Scene Type | Description | Example Use Cases |
|-----------|-------------|-------------------|
| **indoor_small** | Small enclosed spaces | Bedrooms, offices, closets |
| **indoor_large** | Large indoor areas with objects | Warehouses, workshops |
| **outdoor** | Exterior scenes | Streets, parks, parking lots |
| **close_up** | Macro/near-field (<1m) | Product photography, textures |
| **far_away** | Distant objects (>15m) | Buildings, landscapes |
| **glass_shiny** | Reflective surfaces | Windows, mirrors, polished metal |

### Model Recommendation Logic

**Matching Models** (from fast to slow):

**Fast Models**:
- **SIFT** - Universal, scale-invariant
- **LightGlue** - Modern, efficient
- **XFeat** - Robust to lighting
- **xfeat+lightglue** - Enhanced accuracy

**Medium Speed**:
- **SuperGlue** - Handles clutter/occlusions
- **OmniGlue** - Complex scenes
- **XoFTR** - Low texture, reflections
- **RIPE** - Specular reflections

**Slow (High Quality)**:
- **DISK** - Dense features
- **Mast3R** - 3D-aware
- **DUSt3R** - Dense reconstruction

**Recommendation Strategy**:
1. **Indoor Small** → LightGlue (fast, abundant features)
2. **Indoor Large** → SuperGlue (robust to clutter)
3. **Outdoor** → XFeat (scale + lighting invariance)
4. **Close-up** → DISK (dense texture features)
5. **Far-away** → SIFT (reliable at distance)
6. **Glass/Shiny** → XoFTR (handles distortions)

**Speed Preference Options**:
- `fast` - Prioritize speed (SIFT, LightGlue)
- `balanced` - Default, primary recommendation
- `accurate` - Prioritize quality (dense/specialized models)

---

## 🔧 Technical Details

### CLIP Model Configuration
- **Architecture**: ViT-B/32 (Vision Transformer)
- **Parameters**: ~150M
- **Image Resolution**: 224×224
- **Text Encoder**: Transformer
- **Device**: Auto-detect (CUDA if available, else CPU)

### Text Prompt Engineering
- **6 prompts per scene type** = 36 total prompts
- Average embeddings per scene for robustness
- Examples:
  - `"a photo of a small indoor room"`
  - `"a bedroom interior"`
  - `"an office cubicle"`

### Performance
- **Cold start**: ~3 seconds (model loading)
- **Single classification**: 2-5 seconds
- **Batch (10 images)**: ~30-50 seconds
- **Memory**: ~2GB (CLIP model + PyTorch)

---

## 📊 Usage Examples

### Web Interface

**Single Image**:
1. Open `http://localhost:8080`
2. Click "Choose Image" or drag-and-drop
3. View results with model recommendation

**Batch Analysis**:
1. Select multiple images (Ctrl/Cmd + Click)
2. Upload all at once
3. See dataset summary:
   - Scene distribution chart
   - Recommended models for dataset
   - Individual image grid

### Command Line

```bash
# Basic classification
python classify_scene.py office.jpg
# Output: indoor_small (87.3%)

# Verbose output
python classify_scene.py -v bedroom.jpg
# Shows all scene type scores

# JSON for automation
python classify_scene.py --output json dataset/*.jpg > results.json

# CSV for analysis
python classify_scene.py --output csv *.jpg > results.csv
```

---

## 🔌 Integration with 3D Pipelines

### Python Integration

```python
from scene_classifier import SceneClassifier

# Initialize once
classifier = SceneClassifier()

# Classify dataset
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = {}
scene_counts = {}

for img in images:
    scene_type, confidence = classifier.get_top_prediction(img)
    scene_counts[scene_type] = scene_counts.get(scene_type, 0) + 1
    results[img] = (scene_type, confidence)

# Get dominant scene
dominant_scene = max(scene_counts.items(), key=lambda x: x[1])[0]

# Get recommended model
model_config = classifier.get_model_recommendation(dominant_scene)
print(f"Use {model_config['name']} - {model_config['reason']}")

# Apply to reconstruction
if model_config['name'] == 'LightGlue':
    run_reconstruction(images, matcher='lightglue')
elif model_config['name'] == 'SIFT':
    run_reconstruction(images, matcher='sift')
```

### REST API Integration

```bash
# Single image
curl -X POST -F "image=@photo.jpg" http://localhost:8080/api/classify

# Batch processing
curl -X POST \
  -F "images=@img1.jpg" \
  -F "images=@img2.jpg" \
  -F "images=@img3.jpg" \
  http://localhost:8080/api/classify-batch
```

---

## 🛠️ Development

### Requirements
See `requirements_classifier.txt`:
- `torch` - PyTorch for CLIP
- `torchvision` - Image preprocessing
- `ftfy` - Text preprocessing
- `regex` - Pattern matching
- `tqdm` - Progress bars
- `Pillow` - Image handling
- `Flask` - Web server

### Adding New Scene Types

1. **Update `scene_classifier.py`**:
```python
SCENE_TYPES = {
    # ... existing ...
    "new_scene": [
        "a photo of a new scene type",
        "prompt variation 2",
        # ... 6 prompts total
    ]
}

SCENE_DESCRIPTIONS = {
    # ... existing ...
   "new_scene": "Description of new scene"
}

MODEL_RECOMMENDATIONS = {
    # ... existing ...
    "new_scene": {
        "primary": {"name": "ModelName", "speed": "fast", "reason": "..."},
        # ... alternatives
    }
}
```

2. **Restart server** - text embeddings computed on startup

### Customizing Model Recommendations

Edit `MODEL_RECOMMENDATIONS` in `scene_classifier.py`:
- Change `primary` model for any scene type
- Add `alternative`, `fast_option`, or `specialized` variants
- Modify speed ratings: `fast`, `medium`, `slow`

---

## 📝 Sample Output

**Single Image JSON**:
```json
{
  "success": true,
  "image": "data:image/jpeg;base64,...",
  "top_prediction": {
    "type": "outdoor",
    "description": "Outdoor scene - e.g., street, park, parking lot",
    "confidence": 89.2,
    "recommended_model": {
      "name": "XFeat",
      "speed": "fast",
      "reason": "Scale-invariant and robust to lighting changes"
    }
  },
  "all_predictions": [...]
}
```

**Batch JSON**:
```json
{
  "success": true,
  "results": [
    {"filename": "img1.jpg", "scene_type": "outdoor", "confidence": 92.1, ...},
    {"filename": "img2.jpg", "scene_type": "outdoor", "confidence": 87.5, ...}
  ],
  "summary": {
    "total_images": 2,
    "scene_distribution": {
      "outdoor": 100.0
    },
    "dominant_scenes": ["outdoor"],
    "recommended_models": [
      {
        "scene_type": "outdoor",
        "percentage": 100.0,
        "model": {
          "name": "XFeat",
          "speed": "fast",
          "reason": "Scale-invariant and robust to lighting changes"
        }
      }
    ]
  }
}
```

---

## 🐛 Troubleshooting

### "413 Request Entity Too Large"
- **Cause**: Batch upload exceeds 100MB
- **Fix**: Use smaller images or fewer images per batch

### "Model loading failed"
- **Cause**: Missing PyTorch or CLIP dependencies
- **Fix**: `pip install -r requirements_classifier.txt`

### Slow classification
- **Cause**: Running on CPU instead of GPU
- **Check**: `classifier.device` should show `cuda` if GPU available
- **Fix**: Install CUDA-enabled PyTorch

### Port 8080 already in use
- **Fix**: Kill existing process or change port in `web_app.py`
```bash
lsof -ti:8080 | xargs kill -9
```

---

## 📜 License

This project uses OpenAI's CLIP model. See CLIP/LICENSE for details.

---

## 🙏 Credits

- **OpenAI CLIP**: Vision-language model backbone
- **Flask**: Web framework
- **PyTorch**: Deep learning framework

---

**Ready to optimize your 3D reconstruction pipeline! 🚀**
