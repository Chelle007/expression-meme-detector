"""
Hand gesture classification using MediaPipe hand landmarks.
Detects: pointing_up, finger_to_mouth, hands_together, hands_on_head, none.
"""

import numpy as np
import mediapipe as mp
from collections import deque, Counter

mp_hands = mp.solutions.hands


# -----------------------------
# Smoothing (reduces flicker)
# -----------------------------
class GestureSmoother:
    """
    Keeps a short history of predicted labels and returns a stable label.
    - window: how many recent frames to consider
    - min_votes: how many times a label must appear in the window to switch to it
    - hold_frames: once a label is chosen, keep it briefly even if a few frames disagree
    """
    def __init__(self, window=9, min_votes=5, hold_frames=6):
        self.buf = deque(maxlen=window)
        self.min_votes = min_votes
        self.current = "none"
        self.hold = 0
        self.hold_frames = hold_frames

    def update(self, raw_label: str) -> str:
        # Hold current label briefly to avoid rapid switching
        if self.hold > 0 and raw_label != self.current:
            self.hold -= 1
            return self.current

        self.buf.append(raw_label)
        top_label, top_count = Counter(self.buf).most_common(1)[0]

        # Switch only if stable enough
        if top_count >= self.min_votes and top_label != self.current:
            self.current = top_label
            self.hold = self.hold_frames

        return self.current


_smoother = GestureSmoother()


# -----------------------------
# Helpers
# -----------------------------
def _palm_center(hand):
    """Approx palm center from wrist + MCP joints (normalized coords)."""
    pts = [
        hand.landmark[mp_hands.HandLandmark.WRIST],
        hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
        hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
        hand.landmark[mp_hands.HandLandmark.RING_FINGER_MCP],
        hand.landmark[mp_hands.HandLandmark.PINKY_MCP],
    ]
    cx = sum(p.x for p in pts) / len(pts)
    cy = sum(p.y for p in pts) / len(pts)
    return cx, cy


def _fingers_extended(hand_landmarks, margin=0.02):
    """
    Determine finger extension using tip vs PIP with a small margin
    to reduce jitter-based false positives.
    Returns (fingers_up_count, index_only).
    """
    idx_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    idx_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    mid_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    mid_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y

    idx_up = idx_tip < (idx_pip - margin)
    mid_up = mid_tip < (mid_pip - margin)
    ring_up = ring_tip < (ring_pip - margin)
    pinky_up = pinky_tip < (pinky_pip - margin)

    fingers_up = sum([idx_up, mid_up, ring_up, pinky_up])
    index_only = idx_up and (not mid_up) and (not ring_up) and (not pinky_up)
    return fingers_up, index_only, idx_up


# -----------------------------
# Main gesture function
# -----------------------------
def check_gesture(results, face_bbox, frame_shape):
    """
    Detect gesture from MediaPipe hand results.

    Returns one of:
    'pointing_up' | 'finger_to_mouth' | 'hands_together' | 'hands_on_head' | 'none'
    """
    H, W = frame_shape[:2]
    hands = results.multi_hand_landmarks if results and results.multi_hand_landmarks else []
    n_hands = len(hands)

    raw = "none"

    if n_hands == 0:
        return _smoother.update("none")

    # -----------------------------
    # 2 hands
    # -----------------------------
    if n_hands >= 2:
        h1, h2 = hands[0], hands[1]

        # Wrist Y positions for "hands on head"
        w1 = h1.landmark[mp_hands.HandLandmark.WRIST]
        w2 = h2.landmark[mp_hands.HandLandmark.WRIST]
        wy1 = w1.y * H
        wy2 = w2.y * H

        # Hands together: use palm centers (better than wrists)
        c1x, c1y = _palm_center(h1)
        c2x, c2y = _palm_center(h2)
        dx = (c1x - c2x) * W
        dy = (c1y - c2y) * H
        dist = np.sqrt(dx * dx + dy * dy)

        # Threshold (tuneable). 0.35–0.45 are common ranges.
        together = dist < 0.40 * min(W, H)

        # Hands on head: both wrists above face top (with margin)
        if face_bbox is not None:
            x, y, fw, fh = face_bbox
            face_top = y
            margin = 0.15 * fh
            if wy1 < face_top + margin and wy2 < face_top + margin:
                raw = "hands_on_head"
                return _smoother.update(raw)
        else:
            # Fallback if no face bbox
            if wy1 < 0.4 * H and wy2 < 0.4 * H:
                raw = "hands_on_head"
                return _smoother.update(raw)

        if together:
            raw = "hands_together"
            return _smoother.update(raw)

        return _smoother.update("none")

    # -----------------------------
    # 1 hand
    # -----------------------------
    h = hands[0]
    _, index_only, index_up = _fingers_extended(h, margin=0.02)

    idx_tip = h.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    ix, iy = idx_tip.x * W, idx_tip.y * H

    # Finger to mouth: index only + near mouth area (requires face)
    if face_bbox is not None and index_up:
        x, y, fw, fh = face_bbox
        mx, my = x + fw / 2, y + 0.68 * fh
        d = np.sqrt((ix - mx) ** 2 + (iy - my) ** 2)
        if d < 0.32 * min(fw, fh):
            raw = "finger_to_mouth"
            return _smoother.update(raw)

    # Pointing up: index only
    if index_only:
        raw = "pointing_up"
        return _smoother.update(raw)

    return _smoother.update("none")
