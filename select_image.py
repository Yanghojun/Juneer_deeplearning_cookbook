import cv2
from glob import glob
import numpy as np
from tqdm import tqdm
import os

SAVE_DIR = '/home/Juneer_deeplearning_cookbook/data/las_for_autoencoder/'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR + 'original/', exist_ok=True)
os.makedirs(SAVE_DIR + 'noise_deleted/', exist_ok=True)
if __name__ == '__main__':
    img_paths = glob('./data/las_data_annotated/*.png')
    for img_path in tqdm(img_paths[:250], ascii=True):
        img = cv2.imread(img_path)
        cv2.imwrite(SAVE_DIR + 'original/' + img_path.split('/')[-1], img)
        # cv2.imwrite(SAVE_DIR + img_path.split('/')[-1], img)
        # retval, binary_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)     # retval: 사용된 임계값
        binary_img = np.where(img < 120, 0, img)
        result = cv2.hconcat([img, binary_img])
        cv2.imwrite(SAVE_DIR + 'noise_deleted/' + img_path.split('/')[-1], binary_img)
        
    
    # img = cv2.imread('./data/31366_00013.png')
    # mean_denoising = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    # median_blur = cv2.medianBlur(img, 11)
    # gaussian_blur = cv2.GaussianBlur(img, (3, 3), 0)
    # bilateral = cv2.bilateralFilter(img, -1, 10, 5)

    # cv2.imwrite('./data/result/mean_denoising.png', mean_denoising)
    # cv2.imwrite('./data/result/median_blur.png', median_blur)
    # cv2.imwrite('./data/result/gaussian_blur.png', gaussian_blur)
    # cv2.imwrite('./data/result/bilateral.png', bilateral)