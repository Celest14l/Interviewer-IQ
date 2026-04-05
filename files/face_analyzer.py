"""
services/face_analyzer.py
Analyzes a base64-encoded JPEG snapshot (captured every 3-5 s by the frontend).
Uses DeepFace for emotion detection and MediaPipe for gaze + posture.
Falls back to heuristic scores if libraries are not installed.
"""

import base64
import io
import os
import time
import numpy as np
from services.session_store import session_store


def _decode_image(b64: str) -> np.ndarray:
    """Decode base64 image string → numpy BGR array."""
    import cv2
    # Strip data-URL prefix if present
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    img_bytes = base64.b64decode(b64)
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


# ── Emotion detection via DeepFace ─────────────────────────────────────────
def _detect_emotion(img_bgr: np.ndarray) -> tuple[str, float]:
    """Returns (dominant_emotion, confidence 0-10)."""
    try:
        from deepface import DeepFace
        result = DeepFace.analyze(
            img_bgr,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        if isinstance(result, list):
            result = result[0]
        emotions: dict = result.get("emotion", {})
        dominant: str  = result.get("dominant_emotion", "neutral")
        confidence = emotions.get(dominant, 50.0) / 10.0   # 0-100 → 0-10
        return dominant, round(min(confidence, 10.0), 1)
    except ImportError:
        # DeepFace not installed — heuristic
        return "neutral", 5.0
    except Exception:
        return "neutral", 5.0


# ── Gaze & posture via MediaPipe ───────────────────────────────────────────
def _analyze_gaze_posture(img_bgr: np.ndarray) -> dict:
    """Returns gaze_score (0-10), posture_score (0-10), details dict."""
    try:
        import mediapipe as mp
        import cv2

        mp_face_mesh = mp.solutions.face_mesh
        mp_pose     = mp.solutions.pose

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        gaze_score    = 5.0
        posture_score = 5.0
        details       = {}

        # ── Gaze (iris landmarks) ──────────────────────────────────────────
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            res = face_mesh.process(img_rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                # Left iris center (index 468) vs left eye corners (33, 133)
                iris_x = lm[468].x
                eye_left_x  = lm[33].x
                eye_right_x = lm[133].x
                eye_width = abs(eye_right_x - eye_left_x)
                if eye_width > 0:
                    ratio = (iris_x - eye_left_x) / eye_width  # 0.5 = centre
                    deviation = abs(ratio - 0.5)
                    gaze_score = round(max(0, 10 - deviation * 40), 1)
                details["gaze_deviation"] = round(deviation if eye_width > 0 else 0, 3)

        # ── Posture (shoulder alignment) ──────────────────────────────────
        with mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
        ) as pose:
            res = pose.process(img_rgb)
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                # Shoulder tilt (y difference)
                tilt = abs(ls.y - rs.y)
                # Slouch: if shoulders are very high in frame, person is upright
                avg_y = (ls.y + rs.y) / 2
                posture_score = round(max(0, 10 - tilt * 30 - max(0, avg_y - 0.6) * 10), 1)
                details["shoulder_tilt"] = round(tilt, 3)
                details["shoulder_y"]    = round(avg_y, 3)

        return {
            "gaze_score": gaze_score,
            "posture_score": posture_score,
            "details": details,
        }

    except ImportError:
        # MediaPipe not installed — heuristic
        return {"gaze_score": 5.0, "posture_score": 5.0, "details": {"note": "mediapipe_not_installed"}}
    except Exception as e:
        return {"gaze_score": 5.0, "posture_score": 5.0, "details": {"error": str(e)}}


# ── Public function ────────────────────────────────────────────────────────
async def analyze_snapshot(b64_image: str, session_id: str) -> dict:
    """
    Main entry point called from the WebSocket handler.
    Decodes the base64 snapshot, runs emotion + gaze + posture,
    persists result, and returns a summary dict.
    """
    try:
        img = _decode_image(b64_image)
    except Exception as e:
        return {"error": f"Image decode failed: {e}", "emotion": "unknown",
                "gaze_score": 5, "posture_score": 5}

    emotion, emotion_confidence = _detect_emotion(img)
    gaze_posture = _analyze_gaze_posture(img)

    result = {
        "emotion":             emotion,
        "emotion_confidence":  emotion_confidence,
        "gaze_score":          gaze_posture["gaze_score"],
        "posture_score":       gaze_posture["posture_score"],
        "details":             gaze_posture.get("details", {}),
        "ts":                  time.time(),
    }

    # Persist to session
    session_store.append_snapshot(session_id, result)

    # Human-readable feedback
    result["feedback"] = _build_feedback(emotion, gaze_posture["gaze_score"], gaze_posture["posture_score"])
    return result


def _build_feedback(emotion: str, gaze: float, posture: float) -> str:
    tips = []
    if emotion in ("fear", "disgust", "sad"):
        tips.append("Try to appear more confident — take a breath before answering.")
    if gaze < 5:
        tips.append("Maintain eye contact with the camera.")
    if posture < 5:
        tips.append("Sit up straight and keep your shoulders level.")
    return " ".join(tips) if tips else "Looking great — keep it up!"
