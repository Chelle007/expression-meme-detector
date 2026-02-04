# Expression Meme Detector

A facial expression and meme detection system that uses deep learning to classify emotions in images. Built for CSCI218 - Foundations of AI (UOW).

## Features

- **Emotion detection** — Classifies faces into 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Face detection** — Uses OpenCV Haar Cascade for face localisation
- **Hand gesture detection** — MediaPipe for hand landmark detection
- **Pre-trained model** — Emotion classification via ONNX (`emotion_model.onnx`), converted from Keras `emotion_model.hdf5`

## Requirements

- **Python 3.11** (required)
- See `requirements.txt` for package dependencies

## Setup

### 1. Ensure Python 3.11 is installed

Check your Python version:

```bash
python3.11 --version
```

If Python 3.11 is not installed:

- **macOS (Homebrew):** `brew install python@3.11`
- **Ubuntu/Debian:** `sudo apt install python3.11 python3.11-venv`
- **Windows:** Download from [python.org](https://www.python.org/downloads/)

### 2. Create a virtual environment

From the project root directory:

```bash
python3.11 -m venv .venv311
```

This creates a virtual environment named `.venv311` using Python 3.11.

### 3. Activate the virtual environment

**macOS / Linux / WSL (bash/zsh):**

```bash
source .venv311/bin/activate
```

**Windows (Command Prompt):**

```cmd
.venv311\Scripts\activate.bat
```

**Windows (PowerShell):**

```powershell
.venv311\Scripts\Activate.ps1
```

When activated, your prompt will show `(.venv311)` at the start.

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Deactivate (when finished)

```bash
deactivate
```

## Usage

### Run the app (recommended)

1. **One-time: create ONNX emotion model** (if `emotion_model.onnx` is missing):
   ```bash
   pip install -r requirements-convert.txt
   python export_emotion_to_onnx.py
   ```
   This converts `emotion_model.hdf5` → `emotion_model.onnx`. You can then use the main app without TensorFlow.

2. **Run the webcam app:**
   ```bash
   pip install -r requirements.txt
   python main.py
   ```
   Press **q** to quit.

### Optional: run the notebook

1. Open `FT2_Prototype.ipynb` in Jupyter.
2. Select the `.venv311` kernel (install with `pip install ipykernel` and `python -m ipykernel install --user --name=venv311 --display-name="Python 3.11 (.venv311)"` if needed).
3. Run the cells. The notebook still uses Keras; for the app we use ONNX to avoid dependency conflicts.

## Project structure

```
expression-meme-detector/
├── main.py                    # Webcam app (ONNX emotion model)
├── hand_gesture_classifier.py # Hand gesture logic
├── export_emotion_to_onnx.py # One-time: Keras → ONNX
├── emotion_model.hdf5        # Pre-trained Keras model (source)
├── emotion_model.onnx        # ONNX model (create via export_emotion_to_onnx.py)
├── monkey_memes/             # Meme images
├── requirements.txt          # App deps (opencv, mediapipe, onnxruntime)
├── requirements-convert.txt  # One-time conversion deps (tensorflow, tf2onnx)
├── FT2_Prototype.ipynb       # Original notebook (optional)
└── README.md
```

## Troubleshooting

### "Operation not permitted" when activating venv (macOS)

- Grant **Full Disk Access** to Terminal/Cursor in **System Settings → Privacy & Security**
- Or remove quarantine: `xattr -rd com.apple.quarantine .venv311`

### Jupyter kernel not found

```bash
source .venv311/bin/activate
pip install ipykernel
python -m ipykernel install --user --name=venv311 --display-name="Python 3.11 (.venv311)"
```

### Dependency conflicts

The app uses **ONNX Runtime** (not TensorFlow) for emotion inference, so you avoid TensorFlow/MediaPipe/protobuf conflicts. Install with `pip install -r requirements.txt`. Only the one-time conversion script (`export_emotion_to_onnx.py`) needs TensorFlow; use `requirements-convert.txt` for that.
