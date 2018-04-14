import cv2
from comic_sansify import process_frame
from flask import Flask, make_response, request
import numpy as np


app = Flask(__name__)

@app.route('/', methods=['POST'])
def process():
    image_array = np.fromstring(request.data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    image_processed = process_frame(image)
    _, img_encoded = cv2.imencode('.jpg', image_processed)

    response = make_response(img_encoded.tostring())
    response.headers.set('Content-Type', 'image/jpeg')
    response.headers.set('Content-Disposition', 'attachment')

    return response