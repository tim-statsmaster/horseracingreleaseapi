import cv2
import numpy as np

def encode_img(img:np.array):
    img_str = cv2.imencode('.jpg', img)[1].tostring()
    return img_str

def decode_img(img_str:str):
    img = np.fromstring(img_str, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img