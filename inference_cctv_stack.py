from my_utils import stackImages
from utils.hubconf import custom
from utils.plots import plot_one_box
import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--txt", type=str, required=True,
                help="path to RTSP link txt file")
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to best.pt (YOLOv7) model")

args = vars(ap.parse_args())
path_to_txt = args['txt']
path_to_model = args['model']

f = open(path_to_txt, 'r')
rtsp_links =  f.read().splitlines()

# Load Model
color = [[0, 255, 0], [0, 0, 255]]
class_names = ['person', 'on_mobile']

model = custom(path_or_model=path_to_model)


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
        ############
        person_no = []
        mobile_no = []
        results = model(img)
        # Bounding Box
        box = results.pandas().xyxy[0]
        class_list = box['class'].to_list()
        for i, class_id in zip(box.index, class_list):
            xmin, ymin, xmax, ymax = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), int(box['ymax'][i])
            person_no.append(class_id)
            if class_id==1:
                mobile_no.append(class_id)
            bbox = [xmin, ymin, xmax, ymax]
            plot_one_box(bbox, img, label=class_names[class_id], color=color[class_id], line_thickness=2)
        
        # putText
        cv2.putText(img, f'Number of People: {int(len(person_no))}', (40, 50),
                            cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
        cv2.putText(img, f'Mobile Phone Useage: {int(len(mobile_no))}', (40, 100),
                            cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
        ############
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


