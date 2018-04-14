import numpy as np
import cv2

from PIL import Image
import PIL.ImageOps
import pytesseract
import argparse
import os

# TODO class
text_spotter = cv2.text.TextDetectorCNN_create(
    'textbox.prototxt', 'TextBoxes_icdar13.caffemodel')


def get_text_locations(frame):
    rects, probabilities = text_spotter.detect(frame)
    thres = 0.4
    return [
        rect for rect, probability in zip(rects, probabilities)
        if probability > thres
    ]


def draw_rectangles(frame, rectangles, colors=None, filled=False):
    if colors is None:
        colors = [(255, 0, 0)] * len(rectangles)

    new_frame = frame.copy()

    for rect, color in zip(rectangles, colors):
        cv2.rectangle(
            new_frame,
            (rect[0], rect[1]),
            (rect[0] + rect[2], rect[1] + rect[3]),
            color,
            -1 if filled else 2)

    return new_frame


def get_coordinates(frame, rect, padding=0):
    y_max, x_max = frame.shape[:2]
    x, y, width, height = rect

    x_slice = slice(
        max(x - padding, 0), min(x + width + padding, x_max - 1))
    y_slice = slice(
        max(y - padding, 0), min(y + height + padding, y_max - 1))

    return x_slice, y_slice


def cover_rectangles(frame, rectangles):
    if not rectangles:
        return frame

    new_frame = frame.copy()

    def _get_fill_rectangles():
        for rect in rectangles:
            x, y = get_coordinates(frame, rect, padding=10)
            selection = frame[y, x]
            color = [np.median(selection[:, :, dim]) for dim in range(3)]
            yield rect, color

    rectangles, colors = zip(*_get_fill_rectangles())
    new_frame = draw_rectangles(new_frame, rectangles, colors, True)

    for rect in rectangles:
        x, y = get_coordinates(new_frame, rect, padding=15)
        new_frame[y, x] = cv2.blur(new_frame[y, x], (20, 20))

    return new_frame


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
    return pytesseract.image_to_string(gray)


def main():
    cap = cv2.VideoCapture(0)

    latest_text = 'comic sans'

    while True:
        # TODO limit frame rate

        ret, frame = cap.read()
        text_rectangles = get_text_locations(frame)

        text = ocr(frame)
        if text:
            latest_text = text
        print("\"" + latest_text + "\"")
        frame = cover_rectangles(frame, text_rectangles)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
