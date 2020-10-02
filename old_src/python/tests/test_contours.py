#!/usr/bin/env python

import cv2 as cv
from utils import getImageContours, getRect


def testRectContour():
    img = getRect()
    contours, hierarchy = getImageContours(img)
    cv.drawContours(img, contours, -1, (0, 255, 0), 3)

    cv.namedWindow("Rectangle", cv.WINDOW_AUTOSIZE)
    cv.imshow("Rectangle", img)
    cv.waitKey(1)


def testContour():
    cap = cv.VideoCapture(0)
    try:
        while(True):
            ret, frame = cap.read()
            contours, hierarchy = getImageContours(frame)
            cv.drawContours(frame, contours, -1, (0, 255, 0), 3)

            cv.imshow('frame', frame)
            # cv.namedWindow("Contour", cv.WINDOW_AUTOSIZE)
            # cv.imshow("Contour", img)
            cv.waitKey(1)

    except Exception as e:
        print(e)
        pass

    finally:
        cap.release()
        cv.destroyAllWindows()


def main():
    # testRectContour()
    # testContour()


if __name__ == '__main__':
    main()
