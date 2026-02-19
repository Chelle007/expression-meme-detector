# Expression Meme Detector

A facial expression and meme detection system that uses deep learning to classify emotions in images. Built for CSCI218 - Foundations of AI (UOW).

## Features

- **Emotion detection** — Classifies faces into 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Face detection** — Uses OpenCV Haar Cascade for face localisation
- **Hand gesture detection** — MediaPipe for hand landmark detection
- **Pre-trained model** — Emotion classification via ONNX (choose from `FER_models/models/` or `emotion_model.onnx`)

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

### Web app (recommended)

```bash
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5001** in your browser. Allow camera access to see the webcam, emotion probability bars, and meme result.

### CLI webcam app

```bash
python main.py
```

Press **q** to quit.

## Project structure

```
expression-meme-detector/
├── app.py                     # Web app (Flask)
├── static/
│   └── index.html             # Web UI (webcam, emotion bars, meme)
├── main.py                    # CLI webcam app
├── hand_gesture_classifier.py # Hand gesture logic
├── emotion_model.onnx        # Pre-trained emotion model (or in models/)
├── monkey_memes/             # Meme images
├── requirements.txt          # Dependencies (opencv, mediapipe, onnxruntime, flask)
└── README.md
```

## Troubleshooting

### "Operation not permitted" when activating venv (macOS)

- Grant **Full Disk Access** to Terminal/Cursor in **System Settings → Privacy & Security**
- Or remove quarantine: `xattr -rd com.apple.quarantine .venv311`

### Dependency conflicts

The app uses **ONNX Runtime** for emotion inference. Install with `pip install -r requirements.txt`.

### Why only some emotion models load (e.g. only `emotion_model.onnx`)

Models in `FER_models/models/` are discovered automatically, but only models that **load successfully** appear in the dropdown. Common reasons others fail:

1. **Missing external data** — Larger ONNX models (e.g. ResNet, EfficientNet) are often exported with weights in a separate `.onnx.data` file. If that file is not in `FER_models/models/` next to the `.onnx`, loading fails with a "file_size: No such file or directory [*.onnx.data]" error. **Fix:** Add the matching `.onnx.data` file next to the `.onnx`, or use a model that is stored in a single file (e.g. `emotion_model.onnx` or a smaller FER model).
2. **Input shape** — The app currently feeds a fixed 64×64 grayscale face crop. If a model was trained with a different size (e.g. 48×48) or layout, inference may fail at runtime even if the session loads; the dropdown only hides models that fail at **load** time (e.g. missing `.onnx.data`).

When you run `python app.py`, the console prints which models loaded and, for each failure, a short reason (e.g. "Uses external data (missing .onnx.data file)").
