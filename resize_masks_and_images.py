import os
import imageio
import torch
import shutil
import numpy as np
import torchvision.transforms.functional as Tf
from tqdm import tqdm
from skimage import img_as_ubyte
from PIL import Image

in_masks_folder = '/home/csantiago/data/data-inpainting-RePaint/masks'
in_gt_folder = '/home/csantiago/data/data-inpainting-RePaint/masked_images'

out_masks_folder = '/home/csantiago/data/data-inpainting-RePaint/resized_masks'
out_gt_folder = '/home/csantiago/data/data-inpainting-RePaint/resized_masked_images'

for output_folder in [out_masks_folder, out_gt_folder]: 
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

target_size = 256

output_folder = out_masks_folder
fill_value = 255

for input_folder in [in_masks_folder, in_gt_folder]:
    if input_folder == in_gt_folder:
        output_folder = out_gt_folder
        fill_value = 0
        
    for filename in tqdm(os.listdir(input_folder)):
        if not filename.lower().endswith(".png"):
            continue
            
        image_path = os.path.join(input_folder, filename)
        img = Image.open(image_path)

        w, h = img.size
        scale = target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
    
        pad_w = (target_size - new_w) // 2
        pad_h = (target_size - new_h) // 2
        padding = (pad_w, pad_h, target_size - new_w - pad_w, target_size - new_h - pad_h)
        img = Tf.pad(img, padding, fill=fill_value)
    
        output_path = os.path.join(output_folder, filename)
        imageio.imwrite(output_path, img)

