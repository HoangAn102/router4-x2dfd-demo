"""Lazy adapter to the installed DeepfakeBench implementation, without chdir."""
import ast
import importlib
import importlib.util
import inspect
import sys
import textwrap
from functools import lru_cache
from pathlib import Path
from threading import RLock

import numpy as np

from ..config import settings
from ..utils.logger import logger


class FaceCropBackendUnavailable(RuntimeError):
    code = "FACE_CROP_BACKEND_UNAVAILABLE"


_lock = RLock()


def _check_contract(fn):
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    expected = ["face_detector", "predictor", "image", "res", "mask"]
    if ([p.name for p in params] != expected
            or params[3].default != 256 or params[4].default is not None
            or any(p.kind not in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)
                   for p in params[:3])):
        raise FaceCropBackendUnavailable(
            f"Unsupported extract_aligned_face_dlib signature: {sig}; "
            "expected (face_detector, predictor, image, res=256, mask=None). "
            "Inspect the installed preprocess.py; do not substitute an algorithm."
        )
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    align = [n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == "img_align_crop"]
    if len(align) != 1:
        raise FaceCropBackendUnavailable("Cannot verify img_align_crop in installed source")
    args = align[0].args
    defaults = dict(zip([a.arg for a in args.args][-len(args.defaults):], args.defaults))
    value = defaults.get("scale")
    if not isinstance(value, ast.Constant) or value.value != 1.3:
        raise FaceCropBackendUnavailable("Installed img_align_crop scale is not 1.3")


@lru_cache(maxsize=4)
def _load(preprocess_dir, predictor_path):
    """Only successfully initialized runtimes are cached."""
    with _lock:
        try:
            dlib = importlib.import_module("dlib")
            source = Path(preprocess_dir).expanduser().resolve() / "preprocess.py"
            predictor = Path(predictor_path).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(f"Missing preprocess.py: {source}")
            if not predictor.is_file():
                raise FileNotFoundError(f"Missing 81-landmark predictor: {predictor}")
            import hashlib
            name = "_safevision_dfb_" + hashlib.sha256(str(source).encode()).hexdigest()[:16]
            module = sys.modules.get(name)
            if module is None:
                spec = importlib.util.spec_from_file_location(name, source)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load {source}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                try:
                    spec.loader.exec_module(module)
                except BaseException:
                    sys.modules.pop(name, None)
                    raise
            fn = module.extract_aligned_face_dlib
            _check_contract(fn)
            detector = dlib.get_frontal_face_detector()
            shape_predictor = dlib.shape_predictor(str(predictor))
            return fn, detector, shape_predictor
        except Exception as exc:
            if isinstance(exc, FaceCropBackendUnavailable):
                raise
            raise FaceCropBackendUnavailable(f"DeepfakeBench dlib unavailable: {exc}") from exc


def runtime():
    root = Path(settings.DEEPFAKEBENCH_PREPROCESS_DIR).expanduser().resolve()
    predictor = (Path(settings.DLIB_PREDICTOR_PATH).expanduser().resolve()
                 if settings.DLIB_PREDICTOR_PATH else
                 root / "dlib_tools" / "shape_predictor_81_face_landmarks.dat")
    # Serialize cache lookup as well as initialization, avoiding duplicate first loads.
    with _lock:
        return _load(str(root), str(predictor))


@lru_cache(maxsize=16)
def _warn_fallback(reason):
    logger.warning("Mock mode: dlib unavailable; using Haar dev fallback: %s", reason)


def backend():
    selected = settings.FACE_CROP_BACKEND
    if selected == "haar":
        return "haar"
    if selected not in ("auto", "dlib"):
        raise FaceCropBackendUnavailable(f"Unknown FACE_CROP_BACKEND: {selected}")
    try:
        runtime()
    except FaceCropBackendUnavailable as exc:
        if selected == "dlib" or settings.RUN_MODE == "live":
            raise
        _warn_fallback(str(exc))
        return "haar"
    return "dlib"


def aligned_bgr(frame_bgr):
    if settings.FACE_CROP_SIZE != 256 or settings.FACE_CROP_SCALE != 1.3:
        raise FaceCropBackendUnavailable("dlib training contract requires scale=1.3, size=256")
    fn, detector, predictor = runtime()
    # DeepfakeBench receives BGR and performs its own RGB conversion internally.
    with _lock:
        result = fn(detector, predictor, frame_bgr)
    if result is None:
        return None
    if not isinstance(result, tuple) or len(result) != 3:
        raise FaceCropBackendUnavailable("Unexpected DeepfakeBench return contract; expected 3-tuple")
    crop, landmarks, _ = result
    if crop is None:
        return None
    if landmarks is None:
        return None
    if np.asarray(landmarks).shape != (81, 2):
        raise FaceCropBackendUnavailable("Expected 81 landmarks; check predictor identity")
    if not isinstance(crop, np.ndarray) or crop.shape != (256, 256, 3) or crop.dtype != np.uint8:
        raise FaceCropBackendUnavailable("DeepfakeBench output must be uint8 BGR 256x256x3")
    return crop
