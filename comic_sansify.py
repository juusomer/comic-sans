import numpy as np
import cv2
import pytesseract
from PIL import ImageFont, Image, ImageDraw


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


def extract_text(frame, text_rectangles):
    for rect in text_rectangles:
        x, y = get_coordinates(frame, rect, 5)
        text_area = frame[y, x]
        text = ocr(text_area)
        yield text


def add_comic_sans(frame, text_rectangles, strings):
    cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)

    for string, rectangle in zip(strings, text_rectangles):
        x, y, height, width = rectangle
        draw = ImageDraw.Draw(pil_im)
        font = ImageFont.truetype('/Library/Fonts/Comic Sans MS.ttf', int(height / 3))
        draw.text((x, y), string, font=font, fill=(255, 192, 203))

    return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)


def ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return pytesseract.image_to_string(gray)


def process_frame(frame):
    text_rectangles = get_text_locations(frame)
    strings = list(extract_text(frame, text_rectangles))
    frame = cover_rectangles(frame, text_rectangles)
    frame = add_comic_sans(frame, text_rectangles, strings)
    return frame


def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', process_frame(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
