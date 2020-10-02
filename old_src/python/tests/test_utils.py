import cv2 as cv
import numpy as np
import utils as ut


def testBbox(n=10):
    cap = cv.VideoCapture(0)
    img_seq = []
    for i in range(n):
        _, frame = cap.read()
        img_seq.append(frame)
    img_seq = np.stack(img_seq)

    (frames_idx, bbox) = ut.bbox(img_seq)

    n = 0
    for i, img in enumerate(img_seq):
        if i not in frames_idx:
            print('could not find any person in frame number ' + str(i) + '!')
            n += 1
            continue
        b = bbox[i-n]
        (y0, y1, x0, x1) = [int(p) for p in b]
        cut = img[y0:y1, x0:x1]

        ut.imgDisplayCV(cut, "cut")
        cv.waitKey(0)
    return


def testContourAux(createContour):
    cap = cv.VideoCapture(0)
    _, frame = cap.read()
    height, width, _ = frame.shape
    contour = createContour(width, height)

    while True:
        _, frame = cap.read()
        img = ut.drawContour(frame, contour)
        ut.imgDisplayCV(img, "img")
        cv.waitKey(1)


def testCreateDiagonalContour():
    createContour = ut.createDiagonalContour
    testContourAux(createContour)
    return


def testCreateCircleContour():
    createContour = ut.createCircleContour
    testContourAux(createContour)
    return


def testGetRect():
    img = ut.getRect()
    ut.imgDisplayCV(img, "Rectangle")
    cv.waitKey(0)
    return


def testContour(color_image):
    contours, hierarchy = ut.getImageContours(color_image)
    cv.drawContours(color_image, contours, -1, (0, 255, 0), 3)

    cv.namedWindow("Contour", cv.WINDOW_AUTOSIZE)
    cv.imshow('Contour', color_image)
    cv.waitKey(1)
    return


def testShiftImg(color_img):
    dst = ut.shiftImg(color_img, x_shift=320, y_shift=240)
    ut.imgDisplayCV(dst, "shifted image")
    return


def main():
    # testCreateDiagonalContour()
    # testCreateCircleContour()
    # testGetRect()
    testBbox()


if __name__ == '__main__':
    main()
