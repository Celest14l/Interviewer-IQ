"""Standalone face-analysis worker for the Python 3.12 CV runtime."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time

import numpy as np


def decode_image(image_b64: str) -> np.ndarray:
    import cv2

    cleaned = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
    image_bytes = base64.b64decode(cleaned)
    buffer = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    return image


def detect_emotion(image_bgr: np.ndarray) -> tuple[str, float]:
    from deepface import DeepFace

    result = DeepFace.analyze(
        image_bgr,
        actions=["emotion"],
        enforce_detection=False,
        silent=True,
    )
    if isinstance(result, list):
        result = result[0]
    emotions = result.get("emotion", {})
    dominant = result.get("dominant_emotion", "neutral")
    confidence = min(float(emotions.get(dominant, 50.0)) / 10.0, 10.0)
    return dominant, round(confidence, 1)


def analyze_gaze_posture(image_bgr: np.ndarray) -> dict:
    import cv2
    import mediapipe as mp

    solutions = getattr(mp, "solutions", None)
    if solutions is None:
        from mediapipe.python import solutions as mp_solutions

        solutions = mp_solutions

    mp_face_mesh = solutions.face_mesh
    mp_pose = solutions.pose

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gaze_score = 5.0
    posture_score = 5.0
    details: dict = {}

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        face_result = face_mesh.process(rgb)
        if face_result.multi_face_landmarks:
            landmarks = face_result.multi_face_landmarks[0].landmark
            iris_x = landmarks[468].x
            left_x = landmarks[33].x
            right_x = landmarks[133].x
            eye_width = abs(right_x - left_x)
            deviation = 0.0
            if eye_width > 0:
                ratio = (iris_x - left_x) / eye_width
                deviation = abs(ratio - 0.5)
                gaze_score = round(max(0.0, 10.0 - deviation * 40.0), 1)
            details["gaze_deviation"] = round(deviation, 3)

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        pose_result = pose.process(rgb)
        if pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            tilt = abs(left_shoulder.y - right_shoulder.y)
            avg_y = (left_shoulder.y + right_shoulder.y) / 2
            posture_score = round(max(0.0, 10.0 - tilt * 30.0 - max(0.0, avg_y - 0.6) * 10.0), 1)
            details["shoulder_tilt"] = round(tilt, 3)
            details["shoulder_y"] = round(avg_y, 3)

    return {
        "gaze_score": gaze_score,
        "posture_score": posture_score,
        "details": details,
    }


def build_feedback(*, emotion: str, gaze_score: float, posture_score: float) -> str:
    tips: list[str] = []
    if emotion in {"fear", "sad", "disgust"}:
        tips.append("Try to settle your expression and take a breath before answering.")
    if gaze_score < 5:
        tips.append("Maintain steadier eye contact with the camera.")
    if posture_score < 5:
        tips.append("Sit taller and keep your shoulders level.")
    return " ".join(tips) if tips else "Looking good so far. Keep your delivery steady."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-base64", required=True)
    args = parser.parse_args()

    image = decode_image(args.image_base64)
    emotion, emotion_confidence = detect_emotion(image)
    gaze_posture = analyze_gaze_posture(image)
    payload = {
        "emotion": emotion,
        "emotion_confidence": emotion_confidence,
        "gaze_score": gaze_posture["gaze_score"],
        "posture_score": gaze_posture["posture_score"],
        "details": gaze_posture.get("details", {}),
        "feedback": build_feedback(
            emotion=emotion,
            gaze_score=gaze_posture["gaze_score"],
            posture_score=gaze_posture["posture_score"],
        ),
        "ts": time.time(),
    }
    json.dump(payload, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
