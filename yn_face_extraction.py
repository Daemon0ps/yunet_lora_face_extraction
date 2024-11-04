import os
import cv2
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
YUNET_PATH = (
    "./opencv_zoo/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
IMG_TYPES = [
    "blp",
    "bmp",
    "dib",
    "bufr",
    "cur",
    "pcx",
    "dcx",
    "dds",
    "ps",
    "eps",
    "fit",
    "fits",
    "fli",
    "flc",
    "ftc",
    "ftu",
    "gbr",
    "gif",
    "grib",
    "h5",
    "hdf",
    "png",
    "apng",
    "jp2",
    "j2k",
    "jpc",
    "jpf",
    "jpx",
    "j2c",
    "icns",
    "ico",
    "im",
    "iim",
    "tif",
    "tiff",
    "jfif",
    "jpe",
    "jpg",
    "jpeg",
    "mpg",
    "mpeg",
    "mpo",
    "msp",
    "palm",
    "pcd",
    "pxr",
    "pbm",
    "pgm",
    "ppm",
    "pnm",
    "psd",
    "bw",
    "rgb",
    "rgba",
    "sgi",
    "ras",
    "tga",
    "icb",
    "vda",
    "vst",
    "webp",
    "wmf",
    "emf",
    "xbm",
    "xpm",
    "nef",
]
__SB__ = lambda t, d: tqdm(
    total=t,
    desc=d,
    bar_format="{desc}: {percentage:3.0f}%|"
    + "| {n_fmt}/{total_fmt} [elapsed: {elapsed} / Remaining: {remaining}] "
    + "{rate_fmt}{postfix}]",
)
_f = lambda f: [
    str(f)[: len(f) - (str(f)[::-1].find("/")) :].lower(),
    str(f)[
        len(f)
        - (str(f)[::-1].find("/")) : (len(f))
        - 1
        - len(f[-(str(f)[::-1].find(".")) :])
    ],
    str(f)[-(str(f)[::-1].find(".")) :].lower(),
]
_imr = (
    lambda x: (int(abs(x[2] * (x[0] / x[1]))), int(abs(x[3])))
    if x[0] < x[1]
    else (int(abs(x[2])), int(abs(x[2] // (x[0] / x[1]))))
    if x[0] > x[1]
    else (x[2], x[3])
)
_chk_bgra = (
    lambda i: np.uint8(i[::, ::, :-1:]) if np.uint8(i).shape[2] == 4 else np.uint8(i)
)
_IMG_LIST_ = lambda fp, exts: [
    fp + f
    for f in os.listdir(fp[:-1:])
    if os.path.isfile(fp + f) and str(f[-(f[::-1].find(".")) :]).lower() in exts
]


class YuNet:
    def __init__(
        self,
        modelPath,
        inputSize=[320, 320],
        confThreshold=0.6,
        nmsThreshold=0.3,
        topK=5000,
        backendId=0,
        targetId=0,
    ):
        self._modelPath = modelPath
        self._inputSize = tuple(inputSize)
        self._confThreshold = confThreshold
        self._nmsThreshold = nmsThreshold
        self._topK = topK
        self._backendId = backendId
        self._targetId = targetId
        self._model = cv2.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId,
        )

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self._model = cv2.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId,
        )

    def setInputSize(self, input_size):
        self._model.setInputSize(tuple(input_size))

    def infer(self, image: np.uint8):
        faces = self._model.detect(image)
        return np.array([]) if faces[1] is None else faces[1]


BACKEND_TARGET_PAIRS = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA],
    [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16],
    [cv2.dnn.DNN_BACKEND_TIMVX, cv2.dnn.DNN_TARGET_NPU],
    [cv2.dnn.DNN_BACKEND_CANN, cv2.dnn.DNN_TARGET_NPU],
]
backend_id = BACKEND_TARGET_PAIRS[2][0]
target_id = BACKEND_TARGET_PAIRS[2][1]
YN_MODEL: object = None


def yn_check_image(image: np.uint8) -> None:
    try:
        mxsz: int = 320
        image = _chk_bgra(np.uint8(image))
        size = max(np.array(image).shape[0:2])
        pad_x = size - image.shape[1]
        pad_y = size - image.shape[0]
        pad_l = pad_x // 2
        pad_t = pad_y // 2
        yn_img = np.pad(
            image,
            ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        yn_img = np.uint8(
            cv2.resize(
                image,
                _imr([image.shape[1], image.shape[0], mxsz, mxsz]),
                interpolation=cv2.INTER_LANCZOS4,
            )
        )
        YN_MODEL = YuNet(
            modelPath=YUNET_PATH,
            inputSize=yn_img.shape[:2][::-1],
            confThreshold=0.55,
            nmsThreshold=0.45,
            topK=2,
            backendId=backend_id,
            targetId=target_id,
        )
        YN_MODEL.setInputSize(yn_img.shape[:2][::-1])
        fd_res = YN_MODEL.infer(yn_img)
        if len(fd_res) == 0:
            return False
        else:
            return True
    except Exception as e:
        print(e)
        return


def yn_faces(file: str, save_path: str, face_size: list[int], image_type: str) -> None:
    def _save_face_(face: np.uint8, cnt: int):
        c_img = np.uint8(imcb(face))
        c_img = np.uint8(
            cv2.resize(
                c_img,
                _imr([c_img.shape[1], c_img.shape[0], face_size[0], face_size[1]]),
                interpolation=cv2.INTER_LANCZOS4,
            )
        )
        _file_name = (
            f"{save_path}{f[1]}_{str(fcnt).zfill(3)}_{str(cnt).zfill(2)}.{image_type}"
        )
        cv2.imwrite(_file_name, c_img)

    def imcb(image):
        def cb(img: np.array, tol: int = 80) -> list:
            mask = img > tol
            if img.ndim == 3:
                mask = np.array(mask).all(3)
            m, n = mask.shape
            mask0, mask1 = mask.any(0), mask.any(1)
            cs, ce = mask0.argmax(), n - mask0[::-1].argmax()
            rs, re = mask1.argmax(), m - mask1[::-1].argmax()
            return [rs, re, cs, ce]

        imgrey = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        ci = cb(imgrey, tol=40)
        return image[ci[0] : ci[1], ci[2] : ci[3]]

    f = _f(file)
    mxsz: int = 320 * 4
    fcnt, _bxo = (0, 10)
    image = _chk_bgra(np.uint8(cv2.imread(file, cv2.IMREAD_COLOR)))
    im_w, im_h = (image.shape[1], image.shape[0])
    bx = min(image.shape[1], image.shape[0])
    z_img = np.uint8(np.zeros((image.shape[0] + bx // 2, image.shape[1] + bx // 2, 3)))
    z_img[bx // 2 : bx // 2 + im_h, bx // 2 : bx // 2 + im_w] = image
    yn_img = np.uint8(
        cv2.resize(
            z_img,
            _imr([z_img.shape[1], z_img.shape[0], mxsz, mxsz]),
            interpolation=cv2.INTER_LANCZOS4,
        )
    )
    YN_MODEL = YuNet(
        modelPath=YUNET_PATH,
        inputSize=yn_img.shape[:2][::-1],
        confThreshold=0.5,
        nmsThreshold=0.45,
        topK=4,
        backendId=backend_id,
        targetId=target_id,
    )
    YN_MODEL.setInputSize(yn_img.shape[:2][::-1])
    fd_res = YN_MODEL.infer(yn_img)
    x, y, w, h = (0, 0, 0, 0)
    if fd_res is None:
        return
    elif fd_res is not None:
        for res in fd_res:
            bbox = np.asarray(res)[0:4].astype(np.int32)
            x, y, w, h = (bbox[0], bbox[1], bbox[2], bbox[3])
            zy = z_img.shape[1] / yn_img.shape[1]
            zx = z_img.shape[0] / yn_img.shape[0]
            try:
                face_0 = np.uint8(
                    z_img[
                        int(abs((y * zy - bx // _bxo))) : int(
                            abs((y * zy + h * zy + bx // _bxo))
                        ),
                        int(abs((zx * x - bx // _bxo))) : int(
                            abs((zx * x + zx * w + bx // _bxo))
                        ),
                    ]
                ).copy()
                if yn_check_image(face_0):
                    _save_face_(face_0, 1)
            except Exception:
                pass
            try:
                face_1 = np.uint8(
                    z_img[
                        int(abs((y * zy - (bx // _bxo) * 2))) : int(
                            abs((y * zy + h * zy + (bx // _bxo) * 2))
                        ),
                        int(abs((zx * x - (bx // _bxo) * 2))) : int(
                            abs((zx * x + zx * w + (bx // _bxo) * 2))
                        ),
                    ]
                ).copy()
                if yn_check_image(face_1):
                    _save_face_(face_1, 2)
            except Exception:
                pass
            try:
                face_2 = np.uint8(
                    z_img[
                        int(abs((y * zy - bx // _bxo))) : int(
                            abs((y * zy + h * zy + bx // _bxo))
                        ),
                        int(abs((zx * x - (bx // _bxo) * 2))) : int(
                            abs((zx * x + zx * w + (bx // _bxo) * 2))
                        ),
                    ]
                ).copy()
                if yn_check_image(face_2):
                    _save_face_(face_2, 3)
            except Exception:
                pass
            try:
                face_3 = np.uint8(
                    z_img[
                        int(abs((y * zy - (bx // _bxo) * 2))) : int(
                            abs((y * zy + h * zy + (bx // _bxo) * 2))
                        ),
                        int(abs((zx * x - bx // _bxo))) : int(
                            abs((zx * x + zx * w + bx // _bxo))
                        ),
                    ]
                ).copy()
                if yn_check_image(face_3):
                    _save_face_(face_3, 4)
            except Exception:
                pass
            try:
                face_4 = np.uint8(
                    z_img[
                        int(abs((y * zy - bx // _bxo))) : int(
                            abs((y * zy + h * zy + bx // _bxo))
                        ),
                        int(abs((zx * x - (bx // _bxo) * 4))) : int(
                            abs((zx * x + zx * w + (bx // _bxo) * 4))
                        ),
                    ]
                ).copy()
                if yn_check_image(face_4):
                    _save_face_(face_4, 5)
            except Exception:
                pass
            try:
                face_5 = np.uint8(
                    z_img[
                        int(abs((y * zy - (bx // _bxo) * 4))) : int(
                            abs((y * zy + h * zy + (bx // _bxo) * 4))
                        ),
                        int(abs((zx * x - bx // _bxo))) : int(
                            abs((zx * x + zx * w + bx // _bxo))
                        ),
                    ]
                ).copy()
                if yn_check_image(face_5):
                    _save_face_(face_5, 6)
            except Exception:
                pass
            try:
                face_6 = np.uint8(
                    z_img[
                        int(abs((y * zy - (bx // _bxo) * 4))) : int(
                            abs((y * zy + h * zy + (bx // _bxo) * 4))
                        ),
                        int(abs((zx * x - (bx // _bxo) * 4))) : int(
                            abs((zx * x + zx * w + (bx // _bxo) * 4))
                        ),
                    ]
                ).copy()
                if yn_check_image(face_6):
                    _save_face_(face_6, 7)
            except Exception:
                pass
            try:
                face_7 = np.uint8(
                    z_img[
                        int(abs((y * zy - bx // _bxo))) : int(
                            abs((y * zy + h * zy + bx // _bxo))
                        ),
                        int(abs((zx * x - (bx // _bxo) * 6))) : int(
                            abs((zx * x + zx * w + (bx // _bxo) * 6))
                        ),
                    ]
                ).copy()
                if yn_check_image(face_7):
                    _save_face_(face_7, 8)
            except Exception:
                pass
            try:
                face_8 = np.uint8(
                    z_img[
                        int(abs((y * zy - (bx // _bxo) * 6))) : int(
                            abs((y * zy + h * zy + (bx // _bxo) * 6))
                        ),
                        int(abs((zx * x - bx // _bxo))) : int(
                            abs((zx * x + zx * w + bx // _bxo))
                        ),
                    ]
                ).copy()
                if yn_check_image(face_8):
                    _save_face_(face_8, 9)
            except Exception:
                pass
            fcnt = fcnt + 1


_rbs = (
    lambda s: str(s).replace(chr(92), chr(47))
    if s[-1] == chr(47)
    else str(s).replace(chr(92), chr(47)) + chr(47)
)

# file_path=_rbs(fr'E:/LORA/dae/train/pics/')
file_path = _rbs(rf"E:/pic_prog/pics/")
# save_path=_rbs(fr'E:/LORA/dae/train/faces/')
save_path = _rbs(rf"E:/pic_prog/faces/")
width_size, height_size = (640, 640)
image_type = "jpg"

if __name__ == "__main__":
    RES_LIST: list = []
    image_size = (int(width_size), int(height_size))
    BACKEND_TARGET_PAIRS = [
        [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],  # 0
        [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA],  # 1
        [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16],  # 2
        [cv2.dnn.DNN_BACKEND_TIMVX, cv2.dnn.DNN_TARGET_NPU],  # 3
        [cv2.dnn.DNN_BACKEND_CANN, cv2.dnn.DNN_TARGET_NPU],  # 4
    ]
    backend_id, target_id = (BACKEND_TARGET_PAIRS[0][0], BACKEND_TARGET_PAIRS[0][1])
    multiprocessing.freeze_support()
    imglist = _IMG_LIST_(file_path, IMG_TYPES)
    with ThreadPoolExecutor(16) as executor:
        status_bar = tqdm(total=len(imglist), desc=r"Face Extraction")
        futures = [
            executor.submit(yn_faces, file, save_path, image_size, image_type)
            for file in imglist
        ]
        for _ in as_completed(futures):
            status_bar.update(n=1)
