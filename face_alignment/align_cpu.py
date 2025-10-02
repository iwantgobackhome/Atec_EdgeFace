import sys
import os

from face_alignment import mtcnn
import argparse
from PIL import Image
from tqdm import tqdm
import random
from datetime import datetime

# CPU 버전 MTCNN 모델 초기화
mtcnn_model = mtcnn.MTCNN(device='cpu', crop_size=(112, 112))

def add_padding(pil_img, top, right, bottom, left, color=(0,0,0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def get_aligned_face_cpu(image_path, rgb_pil_image=None):
    """
    CPU 버전 MTCNN을 사용한 face alignment
    """
    if rgb_pil_image is None:
        img = Image.open(image_path).convert('RGB')
    else:
        assert isinstance(rgb_pil_image, Image.Image), 'Face alignment module requires PIL image or path to the image'
        img = rgb_pil_image
    # find face
    try:
        bboxes, faces = mtcnn_model.align_multi(img, limit=1)
        face = faces[0] if len(faces) > 0 else None
    except Exception as e:
        print('Face detection Failed due to error.')
        print(e)
        face = None

    return face