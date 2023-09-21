import cv2
import os

from tqdm import tqdm

src_dir = 'C:/Users/ANIRUDH/OneDrive/Desktop/Image Deblurring App/test_data'
images = os.listdir(src_dir)
dst_dir = 'C:/Users/ANIRUDH/OneDrive/Desktop/Image Deblurring App/test_data/gaussian_blurred'

for i, img in tqdm(enumerate(images), total=len(images)):
    img = cv2.imread(f"{src_dir}/{images[i]}")
    # add gaussian blurring
    blur = cv2.GaussianBlur(img, (51, 51), 0)
    cv2.imwrite(f"{dst_dir}/{images[i]}", blur)

print('DONE')