"""
Expression Meme Detector - webcam face emotion + hand gesture → meme.
Run: python main.py
Press 'q' to quit.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort

from hand_gesture_classifier import check_gesture

# ---------------------------------------------------------------------------
# Load models & assets
# ---------------------------------------------------------------------------

# Face Emotion Model (ONNX)
EMOTION_ONNX = "emotion_model.onnx"
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
try:
    emotion_session = ort.InferenceSession(EMOTION_ONNX, providers=["CPUExecutionProvider"])
    emotion_input_name = emotion_session.get_inputs()[0].name
    print("✓ Emotion model (ONNX) loaded.")
except Exception as e:
    emotion_session = None
    emotion_input_name = None
    print(f"Emotion model not loaded ({e}). Run export_emotion_to_onnx.py once to create {EMOTION_ONNX}")

# Face Detection (Haar Cascade)
try:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
except AttributeError:
    cascade_path = None
    possible_paths = [
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            cascade_path = path
            break
    if cascade_path is None or not os.path.exists(cascade_path):
        import urllib.request
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        cascade_path = "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            print("Downloading Haar Cascade file...")
            urllib.request.urlretrieve(url, cascade_path)
            print("Download complete!")

face_cascade = cv2.CascadeClassifier(cascade_path)

# Hand Gesture (MediaPipe)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(
    static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
)
print("✓ MediaPipe initialized successfully!")

# ---------------------------------------------------------------------------
# Load meme images
# ---------------------------------------------------------------------------

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
    print(f"Warning: {filename} not found!")
    return np.zeros((500, 500, 3), dtype=np.uint8)


memes = {
    "pointing": load_meme_image("monkey1.jpeg"),
    "thinking": load_meme_image("monkey2.jpeg"),
    "scheming": load_meme_image("monkey3.jpeg"),
    "shocked": load_meme_image("monkey4.jpeg"),
    "unimpressed": load_meme_image("monkey5.jpeg"),
    "calling": load_meme_image("monkey6.jpeg"),
    "stressed": load_meme_image("monkey7.jpeg"),
}

# ---------------------------------------------------------------------------
# Meme mapping (emotion + gesture → meme image & label)
# ---------------------------------------------------------------------------


def get_meme_result(emotion_name, gesture, memes_dict):
    """
    Strict meme mapping:
    1. pointing_up + happy → monkey1
    2. finger_to_mouth + neutral → monkey2
    3. hands_together + happy → monkey3
    4. hands_together + surprise → monkey4
    5. neutral → monkey5
    6. grab → monkey6
    7. hands_on_head + angry/fear/disgust/surprise → monkey7
    """
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
    if gesture == "grab":
        return memes_dict["calling"], "Calling Monkey!"
    if e == "neutral":
        return memes_dict["unimpressed"], "Unimpressed Monkey."

    return None, "No Meme Match"


# ---------------------------------------------------------------------------
# Webcam loop
# ---------------------------------------------------------------------------


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Starting Webcam... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        current_emotion = "neutral"
        face_bbox = None

        if len(faces) > 0:
            (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            face_bbox = (x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if emotion_session:
                try:
                    roi = gray[y : y + h, x : x + w]
                    roi = cv2.resize(roi, (64, 64))
                    roi = roi.astype(np.float32) / 255.0
                    roi = np.expand_dims(roi, axis=(0, -1))  # (1, 64, 64, 1)
                    preds = emotion_session.run(None, {emotion_input_name: roi})[0][0]
                    current_emotion = emotion_labels[preds.argmax()]
                    cv2.putText(
                        frame,
                        f"Emotion: {current_emotion}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )
                except Exception as e:
                    print(e)

        results = hands_detector.process(rgb_frame)
        current_gesture = check_gesture(results, face_bbox, frame.shape)
        gesture_text = "No Hand" if current_gesture == "none" else f"Gesture: {current_gesture}"

        if results and results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        cv2.putText(
            frame, f"Status: {gesture_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )

        meme_img, meme_label = get_meme_result(current_emotion, current_gesture, memes)
        cv2.putText(
            frame, meme_label, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
        )

        if meme_img is not None:
            cv2.imshow("Meme Result", meme_img)

        cv2.imshow("Webcam Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
