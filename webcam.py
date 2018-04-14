import numpy as np
import cv2


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


def select_rectangle(frame, rect):
    x = slice(rect[2], rect[0])
    y = slice(rect[3], rect[1])
    return frame[x, y]


def cover_rectangles(frame, rectangles):
    if not rectangles:
        return frame

    new_frame = frame.copy()

    def _get_fill_rectangles():
        for rect in rectangles:
            rect_area = select_rectangle(frame, rect)
            color = [np.median(rect_area[:, :, dim]) for dim in range(3)]
            yield rect, color

    rectangles, colors = zip(*_get_fill_rectangles())
    return draw_rectangles(new_frame, rectangles, colors, True)


def add_comic_sans(frame, text_items):
    # TODO
    # replace the original text with comic sans in same size, location,
    # orientation and content
    # https://docs.opencv.org/3.1.0/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
    # bonus: rainbow colors
    return frame


def main():
    cap = cv2.VideoCapture(0)

    while True:
        # TODO limit frame rate

        ret, frame = cap.read()
        text_rectangles = get_text_locations(frame)

        frame = cover_rectangles(frame, text_rectangles)
        #frame = draw_rectangles(frame, text_rectangles)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
