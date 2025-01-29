import cv2
import os
import yaml

from tqdm import tqdm

# Read parameters
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

path = config.get("path", "C:/Users/ANIRUDH/OneDrive/Desktop/Image Deblurring App")

src_dir = path + '/test_data'
images = os.listdir(src_dir)
dst_dir = path + '/gaussian_blurred'

for i, img in tqdm(enumerate(images), total=len(images)):
    img = cv2.imread(f"{src_dir}/{images[i]}")
    # add gaussian blurring
    blur = cv2.GaussianBlur(img, (51, 51), 0)
    cv2.imwrite(f"{dst_dir}/{images[i]}", blur)

print('DONE')