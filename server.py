import cv2
from comic_sansify import process_frame

def read(image_stream):
    img_array = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)

def write(cv2_array):
    pass

# TODO request handler