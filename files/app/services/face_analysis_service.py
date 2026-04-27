"""Face snapshot analysis service built on optional DeepFace and MediaPipe integrations."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.settings import get_settings
from app.db.models.snapshot_event import SnapshotEvent
from app.db.models.user import User
from app.schemas.analysis import SnapshotAnalysisResponse
from app.services.file_storage import save_binary_payload
from app.services.interview_service import get_user_interview


def _decode_image_bytes(image_b64: str) -> tuple[bytes, np.ndarray]:
    """Decode a base64 data URL or bare base64 string into raw bytes and a BGR image."""

    import cv2

    cleaned = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
    image_bytes = base64.b64decode(cleaned)
    buffer = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    return image_bytes, image


def _detect_emotion(image_bgr: np.ndarray) -> tuple[str, float]:
    """Return dominant emotion and a normalized confidence score."""

    try:
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
    except Exception:  # noqa: BLE001
        return "neutral", 5.0


def _analyze_gaze_posture(image_bgr: np.ndarray) -> dict:
    """Estimate gaze and posture scores using MediaPipe where available."""

    try:
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
    except Exception as exc:  # noqa: BLE001
        return {
            "gaze_score": 5.0,
            "posture_score": 5.0,
            "details": {"note": f"face-analysis fallback: {exc}"},
        }


def _build_feedback(*, emotion: str, gaze_score: float, posture_score: float) -> str:
    """Create brief coaching feedback from the current snapshot analysis."""

    tips: list[str] = []
    if emotion in {"fear", "sad", "disgust"}:
        tips.append("Try to settle your expression and take a breath before answering.")
    if gaze_score < 5:
        tips.append("Maintain steadier eye contact with the camera.")
    if posture_score < 5:
        tips.append("Sit taller and keep your shoulders level.")
    return " ".join(tips) if tips else "Looking good so far. Keep your delivery steady."


def _run_cv_worker(image_b64: str) -> dict:
    """Run the dedicated Python 3.12 CV worker and parse its JSON output."""

    settings = get_settings()
    python_path = Path(settings.cv_python_path)
    if not python_path.is_absolute():
        python_path = Path.cwd() / python_path
    worker_path = Path(__file__).resolve().parents[2] / "cv_worker" / "analyze_snapshot.py"
    if not python_path.exists():
        python_path = Path(sys.executable)
    if not python_path.exists():
        raise RuntimeError(f"CV runtime not found at {python_path}")
    if not worker_path.exists():
        raise RuntimeError(f"CV worker script not found at {worker_path}")

    result = subprocess.run(
        [str(python_path), str(worker_path), "--image-base64", image_b64],
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "unknown CV worker failure").strip()
        raise RuntimeError(stderr)
    return json.loads(result.stdout)


async def analyze_face_snapshot(
    db: Session,
    *,
    user: User,
    interview_id: str,
    image_b64: str,
) -> SnapshotAnalysisResponse:
    """Analyze a face snapshot, persist the event, and return coaching-friendly scores."""

    interview = get_user_interview(db, user, interview_id)
    if interview is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Interview not found")

    try:
        image_bytes, image = _decode_image_bytes(image_b64)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not decode image: {exc}") from exc

    _, image_path = save_binary_payload(
        payload=image_bytes,
        namespace=f"{user.id}/snapshots",
        suffix=Path("snapshot.jpg").suffix,
    )
    try:
        payload = _run_cv_worker(image_b64)
        model_name = "deepface+mediapipe(cv-worker)"
    except Exception:
        emotion, emotion_confidence = _detect_emotion(image)
        gaze_posture = _analyze_gaze_posture(image)
        payload = {
            "emotion": emotion,
            "emotion_confidence": emotion_confidence,
            "gaze_score": gaze_posture["gaze_score"],
            "posture_score": gaze_posture["posture_score"],
            "details": gaze_posture.get("details", {}),
            "feedback": _build_feedback(
                emotion=emotion,
                gaze_score=gaze_posture["gaze_score"],
                posture_score=gaze_posture["posture_score"],
            ),
            "ts": time.time(),
        }
        model_name = "deepface+mediapipe"

    event = SnapshotEvent(
        interview_id=interview.id,
        image_path=image_path,
        analysis_payload=payload,
        model_name=model_name,
    )
    db.add(event)
    db.commit()
    db.refresh(event)

    return SnapshotAnalysisResponse(
        **payload,
        snapshot_event_id=event.id,
    )


def list_snapshot_events(db: Session, *, user: User, interview_id: str) -> list[SnapshotEvent]:
    """Return all snapshot events for an owned interview."""

    interview = get_user_interview(db, user, interview_id)
    if interview is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Interview not found")

    result = db.scalars(
        select(SnapshotEvent)
        .where(SnapshotEvent.interview_id == interview.id)
        .order_by(SnapshotEvent.created_at.desc())
    )
    return list(result)


def summarize_snapshot_events(events: list[SnapshotEvent]) -> dict:
    """Compute average coaching scores across stored snapshot events."""

    if not events:
        return {"gaze": 5.0, "posture": 5.0, "emotion_confidence": 5.0}

    gaze = sum(float(item.analysis_payload.get("gaze_score", 5.0)) for item in events) / len(events)
    posture = sum(float(item.analysis_payload.get("posture_score", 5.0)) for item in events) / len(events)
    emotion_confidence = (
        sum(float(item.analysis_payload.get("emotion_confidence", 5.0)) for item in events) / len(events)
    )
    return {
        "gaze": round(gaze, 1),
        "posture": round(posture, 1),
        "emotion_confidence": round(emotion_confidence, 1),
    }
