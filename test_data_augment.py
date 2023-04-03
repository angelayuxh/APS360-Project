import cv2
import os
import glob
import numpy as np
import math

# 20 vegetable classes
vegetable_classes = ['asparagus', 'bell pepper', 'broccoli', 'cabbage', 'carrot', 'celery', 'chilli pepper', 'corn',
                     'cucumber', 'eggplant',
                     'lettuce', 'mushroom', 'onion', 'peas', 'potato', 'pumpkin', 'radish', 'spinach', 'sweet potato',
                     'tomato']
try:
    os.makedirs("aug")
except FileExistsError:
    pass

file_dir = os.getcwd() + "/test/"
aug_dir = os.getcwd() + "/aug/"


for name in vegetable_classes:
    veggie_dir = file_dir + name + "/"
    aug_veggie_dir = aug_dir + name + "/"
    filenames = glob.glob(veggie_dir + "*.jpg") + glob.glob(veggie_dir + "*.jpeg") + glob.glob(veggie_dir + "*.png")

    try:
        os.makedirs(aug_veggie_dir)
    except FileExistsError:
        pass

    num_noise = math.ceil((34 - len(filenames) * 4)/len(filenames))
    for i, filename in enumerate(filenames):
        img = cv2.imread(filename)
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)

        # rotate by 90 degrees
        M = cv2.getRotationMatrix2D(center, 90, 1.0)
        img_90 = cv2.warpAffine(img, M, (w, h))
        cv2.imwrite(aug_veggie_dir + str(i) + "-1.jpeg", img_90)

        # rotate by 180 degrees
        img_180 = cv2.flip(cv2.flip(img, 0), 1)
        cv2.imwrite(aug_veggie_dir + str(i) + "-2.jpeg", img_180)

        # rotate by 270 degrees
        M = cv2.getRotationMatrix2D(center, 270, 1.0)
        img_270 = cv2.warpAffine(img, M, (w, h))
        cv2.imwrite(aug_veggie_dir + str(i) + "-3.jpeg", img_270)

        # Add noise to image
        for j in range(4, num_noise+4):
            mean = 0
            variance = 0.1
            sigma = np.sqrt(variance)
            noise = np.random.normal(mean, sigma, img.shape)
            noisy_img = cv2.add(img, noise, dtype=cv2.CV_8U)
            cv2.imwrite(aug_veggie_dir + str(i) + "-" + str(j) + ".jpeg", noisy_img)
