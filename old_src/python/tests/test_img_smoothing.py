import cv2 as cv
import numpy as np
import utils as ut


def testImgGetMean():
    cap = cv.VideoCapture(0)
    imgs = []
    for i in range(100):
        ret, frame = cap.read()
        imgs.append(frame)
    mean = ut.imgGetMean(imgs)

    ut.imgDisplayCV(imgs[0], "first img")
    ut.imgDisplayCV(mean, "mean img")
    cv.waitKey(0)
    return


def testImgGetFiltered(img=ut.getRect()):
    filtered = ut.imgGetFiltered(img)

    ut.imgDisplayCV(img, "img")
    ut.imgDisplayCV(filtered, "filtered img")
    cv.waitKey(1)
    return


def testImgGetAveraged(img=ut.getRect()):
    averaged = ut.imgGetAveraged(img)

    ut.imgDisplayCV(img, "img")
    ut.imgDisplayCV(averaged, "averaged img")
    cv.waitKey(0)
    return


def testImgGetGaussianBlur(img=ut.getRect()):
    blur = ut.imgGetGaussianBlur(img)

    ut.imgDisplayCV(img, "img")
    ut.imgDisplayCV(blur, "Gaussian Blur img")
    cv.waitKey(0)
    return


def testImgGetMedianBlur(img=ut.getRect()):
    img = cv.imread('data/mona_lisa.png')
    median = ut.imgGetMedianBlur(img)
    print("img == median ? " + str(np.array_equal(img, median)))

    ut.imgDisplayCV(img, "img")
    ut.imgDisplayCV(median, "median img")
    cv.waitKey(0)
    return


def testCompareImgMethods():
    cap = cv.VideoCapture(0)
    _, img = cap.read()

    filtered = ut.imgGetFiltered(img)
    averaged = ut.imgGetAveraged(img)
    blur = ut.imgGetGaussianBlur(img)
    median = ut.imgGetMedianBlur(img)

    ut.imgDisplayCV(img, "img")
    ut.imgDisplayCV(filtered, "filtered img")
    ut.imgDisplayCV(averaged, "averaged img")
    ut.imgDisplayCV(blur, "Gaussian Blur img")
    ut.imgDisplayCV(median, "median img")
    cv.waitKey(0)


def main():
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        
        # testImgGetMean()
        testImgGetFiltered(frame)
        # testImgGetAveraged()
        # testImgGetGaussianBlur()
        # testImgGetMedianBlur()
        # testCompareImgMethods()


if __name__ == '__main__':
    main()
