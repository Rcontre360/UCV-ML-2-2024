import cv2
import numpy as np
import os

def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (28, 28))
            img[img == 255] = 238

            images.append(img)
            break

    return np.array(images)

img_list = load_images('./data/+')
np.set_printoptions(precision=2, suppress=True)
for row in img_list[0]:
    print(row)

