import threading
from typing import Any
import insightface
import numpy as np
from numpy.linalg import norm

import roop.globals
from roop.typing import Frame, Face

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=roop.globals.execution_providers)
            FACE_ANALYSER.prepare(ctx_id=0, det_thresh=0.7, det_size=(256, 256))
    return FACE_ANALYSER


def get_one_face(frame: Frame) -> Any:
    face = get_face_analyser().get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(frame: Frame) -> Any:
    try:
        return get_face_analyser().get(frame)
    except IndexError:
        return None

def is_similar(target_face: Face, found_face: Face) -> bool:
    similarity = np.dot(target_face.embedding, found_face.embedding) / (norm(target_face.embedding) * norm(found_face.embedding))
    if similarity > roop.globals.threshold_value:
        return True
    return False 