import os
import cv2
import shutil
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np


from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gen_path = home + '/demo/GEN'
generalli_path = home + '/demo/mc-generalli/snapshots'

pylab.rcParams['figure.figsize'] = 20, 12
config_file = home + "/configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])


coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)

for gen_file in os.listdir(gen_path):
    car_list = []
    cases = []
    with open(os.path.join(gen_path, gen_file)) as f:
        data = f.readlines()
    for url in data:
        cases.append(url.split('/')[5])
    for case in set(cases):
        case_path = os.path.join(generalli_path, case)
        for sub_case in os.listdir(case_path):
            for type in os.listdir(os.path.join(case_path, sub_case)):
                if 'medium' in type:
                    medium_path = os.path.join(case_path, sub_case, type)
                    for img in os.listdir(medium_path):
                        img_path = os.path.join(medium_path, img)
                        img = cv2.imread(img_path)
                        # image = np.array(img)[:, :, [2, 1, 0]]
                        labels = coco_demo.run_on_opencv_image(img)
                        if 'car' in labels:
                            car_list.append('https://s3.amazonaws.com/' + img_path[45:])

    with open(os.path.join(gen_path, 'CAR_' + gen_file), 'w') as f:
        for url in car_list:
            f.write(url)
            f.write('\n')