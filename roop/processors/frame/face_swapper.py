from typing import Any, List, Callable
import cv2
import insightface
import threading
import pickle, glob, pathlib, shutil
from insightface.app.common import Face as IFace
from tqdm.notebook import tqdm

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, is_similar
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/henryruhs/roop/resolve/main/inswapper_128.onnx'])
    return True


def pre_start() -> bool:
    if roop.globals.source_path.endswith(".pkl"):
        return True
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not get_one_face(cv2.imread(roop.globals.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    global FACE_SWAPPER

    FACE_SWAPPER = None


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)

def read_reference_face(path: str):
    try:
        return get_one_face(cv2.imread(path))
    except Exception as e:
        try:
            with open(path, "rb") as f:
                return IFace(pickle.load(f))
        except Exception as e:
            print("Failed to load checkpoint  : %s", e)
            raise

def process_frame(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    if roop.globals.single_face_in_many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for found_face in many_faces:
                if is_similar(target_face, found_face):
                    temp_frame = swap_face(source_face, found_face, temp_frame)
                    break
    elif roop.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for found_face in many_faces:
                temp_frame = swap_face(source_face, found_face, temp_frame)
    else:
        found_face = get_one_face(temp_frame)
        if found_face:
            temp_frame = swap_face(source_face, found_face, temp_frame)
    return temp_frame


def process_frames(source_path: str, target_face_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_face = read_reference_face(source_path)
    target_face = None
    if target_face_path:
        target_face = get_one_face(cv2.imread(target_face_path))
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face, target_face, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


def process_image(source_path: str, target_face_path: str, output_path: str) -> None:
    target_face = None
    if target_face_path:
        target_face = get_one_face(cv2.imread(target_face_path))
    source_paths = glob.glob(source_path)
    for source_path in tqdm(source_paths):
        if len(source_paths) > 1:
            sp = pathlib.Path(source_path)
            op = pathlib.Path(output_path)
            output_path = str(op.parent.joinpath(f"{sp.stem}.jpg"))
            shutil.copy(target_face_path, output_path)
        source_face = read_reference_face(source_path)
        result = process_frame(source_face, target_face, cv2.imread(output_path))
        cv2.imwrite(output_path, result)
        # print(output_path)


def process_video(source_path: str, target_face_path: str, temp_frame_paths: List[str]) -> None:
    roop.processors.frame.core.process_video(source_path, target_face_path, temp_frame_paths, process_frames)
