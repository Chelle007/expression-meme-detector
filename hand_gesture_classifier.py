"""
Hand gesture classification using MediaPipe hand landmarks.
Detects: pointing_up, finger_to_mouth, hands_together, hands_on_head, none.
"""

import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands


def _fingers_extended(hand_landmarks):
    """Count extended fingers (tip higher than PIP). Returns (count, index_only)."""
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
    fingers_up = sum([
        index_tip < index_pip,
        middle_tip < middle_pip,
        ring_tip < ring_pip,
        pinky_tip < pinky_pip,
    ])
    index_only = (
        index_tip < index_pip
        and middle_tip > middle_pip
        and ring_tip > ring_pip
        and pinky_tip > pinky_pip
    )
    return fingers_up, index_only


def check_gesture(results, face_bbox, frame_shape):
    """
    Detect gesture from MediaPipe hand results.

    Returns: 'pointing_up' | 'finger_to_mouth' | 'hands_together' | 'hands_on_head' | 'none'

    Strict mapping:
    1. 1 hand, index up + happy → monkey1
    2. 1 hand, index to mouth + neutral → monkey2
    3. 2 hands together + happy → monkey3
    4. 2 hands together + surprised → monkey4
    5. neutral (no other match) → monkey5
    6. 2 hands on head + angry/fear/disgust/surprise → monkey6
    """
    H, W = frame_shape[:2]
    hands = results.multi_hand_landmarks if results and results.multi_hand_landmarks else []
    n_hands = len(hands)

    if n_hands == 0:
        return "none"

    # --- 2 HANDS ---
    if n_hands == 2:
        h1, h2 = hands[0], hands[1]
        w1 = h1.landmark[mp_hands.HandLandmark.WRIST]
        w2 = h2.landmark[mp_hands.HandLandmark.WRIST]
        wx1, wy1 = w1.x * W, w1.y * H
        wx2, wy2 = w2.x * W, w2.y * H
        dist = np.sqrt((wx1 - wx2) ** 2 + (wy1 - wy2) ** 2)
        together = dist < 0.25 * min(W, H)

        # Hands on head: both wrists above face (or in upper part of frame if no face)
        if face_bbox is not None:
            x, y, w, h = face_bbox
            face_top = y
            # Allow wrists slightly below face top (bbox jitter / different head sizes)
            margin = 0.15 * h
            if wy1 < face_top + margin and wy2 < face_top + margin:
                return "hands_on_head"
        else:
            # Fallback when face not detected: both wrists in upper ~40% of frame
            upper_threshold = 0.4 * H
            if wy1 < upper_threshold and wy2 < upper_threshold:
                return "hands_on_head"
        if together:
            return "hands_together"
        return "none"

    # --- 1 HAND ---
    h = hands[0]
    fingers_up, index_only = _fingers_extended(h)
    idx_tip = h.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    ix, iy = idx_tip.x * W, idx_tip.y * H

    # Finger to mouth: index extended, tip near mouth (need face)
    if face_bbox is not None and index_only:
        x, y, w, h = face_bbox
        mx, my = x + w / 2, y + 0.7 * h
        d = np.sqrt((ix - mx) ** 2 + (iy - my) ** 2)
        if d < 0.25 * min(w, h):
            return "finger_to_mouth"

    # Index pointing up only
    if index_only:
        return "pointing_up"

    return "none"
