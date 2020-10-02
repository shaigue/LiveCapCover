import config
import cv2

from lib.image_processing.silhouette import LivecapBGS

if __name__ == "__main__":
    bgs = LivecapBGS(config.background_path)

    image = list(config.frame_path.glob('*'))[0]
    for image in config.frame_path.glob('*'):
        print(image.name)
        image = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        sub = bgs.process(image)
        cv2.imshow('bg sub', sub)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
