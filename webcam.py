import numpy as np
import cv2

from PIL import Image
import PIL.ImageOps  
import pytesseract
import argparse
import os

# TODO class
text_spotter = cv2.text.TextDetectorCNN_create('textbox.prototxt', 'TextBoxes_icdar13.caffemodel')


def extract_text(frame):
    rects, probabilities = text_spotter.detect(frame)
    vis = frame.copy()
    thres = 0.4

    for r in range(np.shape(rects)[0]):
        if probabilities[r] > thres:
            rect = rects[r]
            cv2.rectangle(vis, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 2)

    return vis


def hide_text(frame, text_items):
    # TODO
    # hide the text items by e.g. selecting stuff from their surroundings,
    # taking the mean color and drawing a rectangle (possibly blurred borders etc.?)
    return frame


def add_comic_sans(frame, text_items):
    # TODO
    # replace the original text with comic sans in same size, location,
    # orientation and content
    # https://docs.opencv.org/3.1.0/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
    # bonus: rainbow colors
    return frame

def ocr(image):
    # load the example image and convert it to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # threshold the image
    gray = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # use pytesseract ocr
    text = pytesseract.image_to_string(gray)
    print(text)
    return text


def main():
    cap = cv2.VideoCapture(0)

    while True:
        # TODO limit frame rate

        ret, frame = cap.read()

        ocr(frame)

        frame = extract_text(frame)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()