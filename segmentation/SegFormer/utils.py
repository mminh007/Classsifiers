import numpy as np
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt

def mapping_color(image:Image):
    """
    Read annotation.xml to retrieve color information for each label.
    """

    color2id = {
            (184, 61, 245): 5, # #b83df5: backgroud
            (255, 53, 94): 4, # #ff355e: road_sign
            (255, 204, 51): 3, # #ffcc33: car
            (221, 255, 51): 2, # #ddff33: marking
            (61,61, 245): 1, # #3d3df5: road_surface 
            } 
    
    img = np.array(image)

    height, width, _ = img.shape
    matrix = np.full(shape = (height, width), fill_value = 0,
                    dtype = np.int32)
    
    for h in range(height):
        for w in range(width):
            color_pixel = tuple(img[h,w,:])

            if color_pixel in color2id:
                matrix[h,w] = color2id[color_pixel]

    return matrix 


def process_data(path_mask = "content/masks", num = 20):
    """
    path_mask = "/content/masks"
    path_image = "/content/images"
    train_path = "/content/datasets/train"
    validation_path = "/content/datasets/validation"
    num: num of images split to training set
    """

    path_image = "/content/images"
    train_path = "/content/datasets/train"
    validation_path = "/content/datasets/validation" 
    for i,f in enumerate(os.listdir(path_mask)):
        if i <= num:
            os.rename(os.path.join(path_image, f), os.path.join(train_path, f))
            os.rename(os.path.join(path_mask, f), os.path.join(train_path, f.replace(".png", "_mask.png")))
        else:
            os.rename(os.path.join(path_image, f), os.path.join(validation_path, f))
            os.rename(os.path.join(path_mask, f), os.path.join(validation_path, f.replace(".png", "_mask.png")))


def show_image(image, mask, alpha_1=1, alpha_2=0.7):
  plt.imshow(image, alpha=alpha_1)
  plt.imshow(mask, alpha=alpha_2, cmap="gray")
  plt.axis("off")

def show_images(images, masks, grid =[1,5]):
  n_rows, n_cols = grid[0], grid[1]
  num_images = n_rows * n_cols
  plt.figure(figsize= (n_rows * 20, n_cols * 20))

  i = 1
  for image, mask in zip(images, masks):
    plt.subplot(n_rows, n_cols, i)
    show_image(image, mask)
    i += 1
    if i > num_images:
      break
    #plt.show()
