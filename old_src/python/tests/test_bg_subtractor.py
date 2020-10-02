#!/usr/bin/env python

import cv2 as cv


def testBS():
    cap = cv.VideoCapture(0)

    fgbg = cv.createBackgroundSubtractorMOG2()

    try:
        while(True):
            ret, frame = cap.read()

            fgmask = fgbg.apply(frame)

            cv.imshow('frame', fgmask)
            cv.waitKey(1)
            # k = cv.waitKey(30) & 0xff
            # if k == 27:
            #     break

    except Exception as e:
        print(e)
        pass

    finally:
        cap.release()
        cv.destroyAllWindows()


def main():
    testBS()


if __name__ == '__main__':
    main()
