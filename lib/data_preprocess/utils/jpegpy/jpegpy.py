#!/usr/bin/env mdl
import cv2
import numpy as np
from simplejpeg import decode_jpeg,encode_jpeg
# from . import _jpegpy


def jpeg_encode(img: np.array, quality=80):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return encode_jpeg(img, quality)


def jpeg_decode(code: bytes):
    img = decode_jpeg(code)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# vim: ts=4 sw=4 sts=4 expandtab
