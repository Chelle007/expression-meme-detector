"""
Expression Meme Detector - Web app.
Run: python app.py
Open http://127.0.0.1:5001 in your browser.
"""

import os
import base64
import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
from flask import Flask, request, jsonify, send_from_directory
from hand_gesture_classifier import check_gesture

app = Flask(__name__, static_folder="static", static_url_path="")

# ---------------------------------------------------------------------------
# Load models & assets
# ---------------------------------------------------------------------------

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Emotion models from FER_models/models (lazy-loaded by filename)
FER_MODELS_DIR = os.path.join("FER_models", "models")
_emotion_sessions = {}  # model_filename -> (session, input_name, input_spec)
_default_model = None

# ImageNet normalization (used by FER ResNet/Custom/MobileNet/EfficientNet)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _get_available_models():
    """Return list of .onnx filenames in FER_models/models."""
    if not os.path.isdir(FER_MODELS_DIR):
        return []
    return sorted(
        f for f in os.listdir(FER_MODELS_DIR)
        if f.endswith(".onnx")
    )


def _get_emotion_session(model_filename):
    """Get or create ONNX session for the given model filename. Returns (session, input_name, input_spec) or (None, None, None)."""
    global _default_model
    if model_filename in _emotion_sessions:
        return _emotion_sessions[model_filename]
    path = os.path.join(FER_MODELS_DIR, model_filename)
    if not os.path.exists(path) and model_filename == "emotion_model.onnx":
        for fallback in ["emotion_model.onnx", os.path.join("models", "emotion_model.onnx")]:
            if os.path.exists(fallback):
                path = fallback
                break
    if not os.path.exists(path):
        print(f"Emotion model {model_filename} not found at {path}")
        return (None, None, None)
    # Use absolute path so ONNX Runtime resolves external .onnx.data relative to the model file
    path = os.path.abspath(path)
    try:
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp = session.get_inputs()[0]
        input_name = inp.name
        # Parse shape: NCHW (batch, C, H, W) or NHWC; dynamic dims can be int 0 or string
        def _dim(d, default):
            try:
                return int(d) if d is not None and str(d).isdigit() else default
            except (TypeError, ValueError):
                return default
        shape = list(inp.shape)
        if len(shape) != 4:
            input_spec = {"channels": 1, "height": 64, "width": 64, "layout": "nchw"}
        else:
            s1, s2, s3, s4 = _dim(shape[0], 1), _dim(shape[1], 64), _dim(shape[2], 64), _dim(shape[3], 1)
            if s2 == 3:
                # NCHW: (batch, 3, H, W)
                channels, height, width, layout = 3, s3, s4, "nchw"
            elif s4 == 3:
                channels, height, width, layout = 3, s2, s3, "nhwc"
            elif s2 == 1:
                channels, height, width, layout = 1, s3, s4, "nchw"
            else:
                # (1, 64, 64, 1) NHWC legacy
                channels, height, width, layout = 1, s2, s3, "nhwc"
            input_spec = {"channels": channels, "height": height, "width": width, "layout": layout}
        _emotion_sessions[model_filename] = (session, input_name, input_spec)
        if _default_model is None:
            _default_model = model_filename
        return (session, input_name, input_spec)
    except Exception as e:
        err_msg = str(e).lower()
        if ".onnx.data" in err_msg or "external" in err_msg or "file_size" in err_msg or "no such file" in err_msg:
            print(f"  [{model_filename}] Uses external data (missing .onnx.data file). Add the .onnx.data file next to the .onnx in FER_models/models/, or use another model.")
        else:
            print(f"  [{model_filename}] {type(e).__name__}: {e}")
        return (None, None, None)


# At startup: try loading each model and report which work / which fail (so user sees why only some load)
_available = _get_available_models()
_loadable_at_startup = []
print("Emotion models in FER_models/models:")
for name in _available:
    session, _, _ = _get_emotion_session(name)
    if session is not None:
        _loadable_at_startup.append(name)
if not _loadable_at_startup:
    for legacy in ["emotion_model.onnx", os.path.join("models", "emotion_model.onnx")]:
        if os.path.exists(legacy):
            _get_emotion_session("emotion_model.onnx")
            _default_model = "emotion_model.onnx"
            _loadable_at_startup = ["emotion_model.onnx"]
            break
if _loadable_at_startup:
    print("✓ Loadable emotion model(s):", ", ".join(_loadable_at_startup))
else:
    print("No emotion model could be loaded.")

# Face cascade
try:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
except AttributeError:
    cascade_path = None
    for path in [
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml"),
    ]:
        if os.path.exists(path):
            cascade_path = path
            break
    if not cascade_path or not os.path.exists(cascade_path):
        import urllib.request
        cascade_path = "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
                cascade_path,
            )
face_cascade = cv2.CascadeClassifier(cascade_path)

# MediaPipe hands
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
)
print("✓ MediaPipe initialized.")

# Memes
FOLDER_NAME = "monkey_memes"


def load_meme_image(filename):
    path = os.path.join(FOLDER_NAME, filename)
    if not os.path.exists(path):
        base_path = os.path.splitext(path)[0]
        for ext in [".png", ".jpg", ".jpeg"]:
            alt_path = base_path + ext
            if os.path.exists(alt_path):
                path = alt_path
                break
    if os.path.exists(path):
        return cv2.imread(path)
    return np.zeros((500, 500, 3), dtype=np.uint8)


memes = {
    "pointing": load_meme_image("monkey1.jpeg"),
    "thinking": load_meme_image("monkey2.jpeg"),
    "scheming": load_meme_image("monkey3.jpeg"),
    "shocked": load_meme_image("monkey4.jpeg"),
    "unimpressed": load_meme_image("monkey5.jpeg"),
    "stressed": load_meme_image("monkey6.jpeg"),
}


def get_meme_result(emotion_name, gesture, memes_dict):
    e = emotion_name.lower()
    if gesture == "hands_on_head" and e in ["angry", "fear", "disgust", "surprise"]:
        return memes_dict["stressed"], "Stressed Monkey!"
    if gesture == "hands_together" and e == "happy":
        return memes_dict["scheming"], "Scheming Monkey!"
    if gesture == "hands_together" and e == "surprise":
        return memes_dict["shocked"], "Shocked Monkey!"
    if gesture == "pointing_up" and e == "happy":
        return memes_dict["pointing"], "Pointing Monkey!"
    if gesture == "finger_to_mouth" and e == "neutral":
        return memes_dict["thinking"], "Thinking Monkey..."
    if e == "neutral":
        return memes_dict["unimpressed"], "Unimpressed Monkey."
    return None, "No Meme Match"


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def process_frame(frame_bgr, model_filename=None):
    """Run face + emotion + hand + meme pipeline. Returns dict for JSON.
    model_filename: optional .onnx filename from FER_models/models; uses default if omitted.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    H, W = frame_bgr.shape[:2]

    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    current_emotion = "Neutral"
    face_bbox = None
    emotion_probs = {label: 0.0 for label in EMOTION_LABELS}
    emotion_probs["Neutral"] = 1.0

    emotion_session, emotion_input_name, input_spec = (None, None, None)
    if model_filename:
        emotion_session, emotion_input_name, input_spec = _get_emotion_session(model_filename)
    if emotion_session is None and _default_model:
        emotion_session, emotion_input_name, input_spec = _get_emotion_session(_default_model)

    if len(faces) > 0:
        (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        face_bbox = (x, y, w, h)
        if emotion_session and emotion_input_name and input_spec:
            try:
                ch, h, w = input_spec["channels"], input_spec["height"], input_spec["width"]
                layout = input_spec["layout"]
                if ch == 3:
                    # RGB: crop from BGR frame, resize, ImageNet normalize, NCHW
                    roi = frame_bgr[y : y + h, x : x + w]
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    roi = cv2.resize(roi, (w, h))
                    roi = roi.astype(np.float32) / 255.0
                    roi = (roi - IMAGENET_MEAN) / IMAGENET_STD
                    roi = np.transpose(roi, (2, 0, 1))  # HWC -> CHW
                    roi = np.expand_dims(roi, axis=0).astype(np.float32)  # (1, 3, H, W)
                else:
                    # Grayscale
                    roi = gray[y : y + h, x : x + w]
                    roi = cv2.resize(roi, (w, h))
                    roi = roi.astype(np.float32) / 255.0
                    if layout == "nchw":
                        roi = np.expand_dims(roi, axis=(0, 1))  # (1, 1, H, W)
                    else:
                        roi = np.expand_dims(roi, axis=(0, -1))  # (1, H, W, 1)
                logits = emotion_session.run(
                    None, {emotion_input_name: roi}
                )[0][0]
                probs = softmax(logits)
                current_emotion = EMOTION_LABELS[int(np.argmax(probs))]
                emotion_probs = {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))}
            except Exception:
                pass

    results = hands_detector.process(rgb)
    gesture = check_gesture(results, face_bbox, frame_bgr.shape)
    meme_img, meme_label = get_meme_result(current_emotion, gesture, memes)

    # Hand landmarks in pixel coords (same as main.py draw_landmarks)
    hand_landmarks_list = []
    if results and results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            points = [
                {"x": int(lm.x * W), "y": int(lm.y * H)}
                for lm in hand_lms.landmark
            ]
            hand_landmarks_list.append(points)
    hand_connections = [list(pair) for pair in mp_hands.HAND_CONNECTIONS]

    out = {
        "emotion_probs": emotion_probs,
        "emotion": current_emotion,
        "gesture": gesture,
        "meme_label": meme_label,
        "frame_width": W,
        "frame_height": H,
        "face_bbox": (
            {"x": int(face_bbox[0]), "y": int(face_bbox[1]), "w": int(face_bbox[2]), "h": int(face_bbox[3])}
            if face_bbox else None
        ),
        "hand_landmarks": hand_landmarks_list,
        "hand_connections": hand_connections,
    }
    if meme_img is not None:
        _, buf = cv2.imencode(".png", meme_img)
        out["meme_image_base64"] = base64.b64encode(buf).decode("utf-8")
    else:
        out["meme_image_base64"] = None
    return out


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/models", methods=["GET"])
def list_models():
    """Return list of emotion model filenames that actually load (skips models missing .onnx.data etc.)."""
    all_models = _get_available_models()
    loadable = []
    for name in all_models:
        session, _, _ = _get_emotion_session(name)
        if session is not None:
            loadable.append(name)
    # Only use default if it actually loaded (is in loadable)
    default = _default_model if _default_model and _default_model in loadable else (loadable[0] if loadable else None)
    return jsonify({"models": loadable, "default": default})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files and "image" not in request.json:
        return jsonify({"error": "No image provided"}), 400
    try:
        model_name = None
        if request.files.get("image"):
            file = request.files["image"]
            buf = np.frombuffer(file.read(), dtype=np.uint8)
            model_name = request.form.get("model") or request.form.get("model_name")
        else:
            data = request.get_json()
            b64 = data.get("image", "").split(",")[-1] if isinstance(data.get("image"), str) else data.get("image")
            buf = base64.b64decode(b64)
            buf = np.frombuffer(buf, dtype=np.uint8)
            model_name = data.get("model") or data.get("model_name")
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid image"}), 400
        result = process_frame(img, model_filename=model_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
