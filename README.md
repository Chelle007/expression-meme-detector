# Expression Meme Detector

A facial expression and meme detection system that uses deep learning to classify emotions in images. Built for CSCI218 - Foundations of AI (UOW).

## Features

- **Emotion detection** — Classifies faces into 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Face detection** — Uses OpenCV Haar Cascade for face localisation
- **Hand gesture detection** — MediaPipe for hand landmark detection
- **Pre-trained model** — Emotion classification via ONNX (`emotion_model.onnx`)

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

Run the webcam app:

```bash
python main.py
```

Press **q** to quit.

## Project structure

```
expression-meme-detector/
├── main.py                    # Webcam app
├── hand_gesture_classifier.py # Hand gesture logic
├── emotion_model.onnx        # Pre-trained emotion model (ONNX)
├── monkey_memes/             # Meme images
├── requirements.txt          # Dependencies (opencv, mediapipe, onnxruntime)
└── README.md
```

## Troubleshooting

### "Operation not permitted" when activating venv (macOS)

- Grant **Full Disk Access** to Terminal/Cursor in **System Settings → Privacy & Security**
- Or remove quarantine: `xattr -rd com.apple.quarantine .venv311`

### Dependency conflicts

The app uses **ONNX Runtime** for emotion inference. Install with `pip install -r requirements.txt`.
