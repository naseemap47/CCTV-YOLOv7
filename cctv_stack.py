from my_utils import stackImages
import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--txt", type=str, required=True,
                help="path to RTSP link txt file")

args = vars(ap.parse_args())
path_to_txt = args['txt']

f = open(path_to_txt, 'r')
rtsp_links =  f.read().splitlines()

camera_len = len(rtsp_links)
if camera_len%3==0:
    empty_img_no=0
else:
    empty_img_no = 3 - camera_len%3

imageBlank = np.zeros((1440, 1200, 3), np.uint8)

caps = []
for link in rtsp_links:
    cap = cv2.VideoCapture(link)
    caps.append(cap)

while True:
    imgs = []
    for cam in caps:
        success, img = cam.read()
        if not success:
            img = imageBlank
            print('[INFO] Camera NOT Working!')
            continue
        imgs.append(img)

    if empty_img_no>0:
        for i in range(0, empty_img_no):
            imgs.append(imageBlank)

    if len(imgs)>0:
        rows = int(len(imgs)/3)
        if len(imgs)>3:
            imgs = np.reshape(imgs, (rows, 3)).tolist()
            stack_img = stackImages(0.2, (imgs))
        else:
            stack_img = stackImages(0.2, (imgs))

        cv2.imshow('CCTV', stack_img)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            cv2.destroyAllWindows()
            break


