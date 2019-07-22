import os
import cv2
import numpy as np

path = "./Monet/Original/"

path = [os.path.join(path,  x) for x in os.listdir(path)]

output_path = "./Monet/"

for p in path:

    im = cv2.imread(p)
    h, w, c = np.shape(im)
    if h < w:
        scale = w//h
        im = cv2.resize(im, (512, 512 * scale))
        n_h, n_w,_ = np.shape(im)
    else :
        scale = h//w
        im = cv2.resize(im, (512 * scale, 512))
        n_h, n_w,_ = np.shape(im)
    print(h, w)
    c_y, c_x = n_h//2, n_w//2
    crop_im = im[c_y - 256: c_y + 256, c_x-256: c_x+256, :]
    cv2.imwrite(output_path + p.split("/")[-1], crop_im)
    
