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

# Emotion model (ONNX) - try root then models/
emotion_onnx_path = "emotion_model.onnx"
if not os.path.exists(emotion_onnx_path):
    emotion_onnx_path = os.path.join("models", "emotion_model.onnx")
emotion_session = None
emotion_input_name = None
try:
    emotion_session = ort.InferenceSession(
        emotion_onnx_path, providers=["CPUExecutionProvider"]
    )
    emotion_input_name = emotion_session.get_inputs()[0].name
    print("✓ Emotion model (ONNX) loaded.")
except Exception as e:
    print(f"Emotion model not loaded: {e}")

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


def process_frame(frame_bgr):
    """Run face + emotion + hand + meme pipeline. Returns dict for JSON."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    H, W = frame_bgr.shape[:2]

    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    current_emotion = "Neutral"
    face_bbox = None
    emotion_probs = {label: 0.0 for label in EMOTION_LABELS}
    emotion_probs["Neutral"] = 1.0

    if len(faces) > 0:
        (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        face_bbox = (x, y, w, h)
        if emotion_session:
            try:
                roi = gray[y : y + h, x : x + w]
                roi = cv2.resize(roi, (64, 64))
                roi = roi.astype(np.float32) / 255.0
                roi = np.expand_dims(roi, axis=(0, -1))
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


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files and "image" not in request.json:
        return jsonify({"error": "No image provided"}), 400
    try:
        if request.files.get("image"):
            file = request.files["image"]
            buf = np.frombuffer(file.read(), dtype=np.uint8)
        else:
            data = request.get_json()
            b64 = data.get("image", "").split(",")[-1] if isinstance(data.get("image"), str) else data.get("image")
            buf = base64.b64decode(b64)
            buf = np.frombuffer(buf, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid image"}), 400
        result = process_frame(img)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
