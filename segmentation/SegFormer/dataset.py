from torch.utils.data import Dataset
import os
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import mapping_color

  

class SegmenDataset(Dataset):
    def __init__(self, path, image_processor, types = "train"):
        self.types = types
        self.path = path
        self.root_dir = os.path.join(self.path + "/" + self.types)
        self.path_list = [f for f in os.listdir(self.root_dir) if "_mask.png" in f] 
        self.image_processor = image_processor
        # self.transform = A.Compose([
        #                     A.RandomCrop(width=512, height=512),
        #                     A.HorizontalFlip(p=0.5),
        #                     A.RandomBrightnessContrast(p=0.2),
        #                     ToTensorV2(),
        #                     ])
        self.mapping = mapping_color

    def __len__(self):
        return len(self.path_list)
    
    def __getitem__(self, idx):
        mask_path  = self.path_list[idx]
        image_path = mask_path.replace("_mask.png", ".png")

        image = Image.open(os.path.join(self.root_dir,image_path))
        mask = Image.open(os.path.join(self.root_dir, mask_path)).convert("RGB")
        mask = self.mapping(mask)  # convert mask.png shape (H,W,3) to (H,W) 
        segmentation_map = Image.fromarray(mask) 

        encoded_inputs = self.image_processor(image, segmentation_map, return_tensors = "pt")
        
        for k,v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()
        # transformed = self.transform(image=image, mask=mask)
        # transformed_image = transformed['image']
        # transformed_mask = transformed['mask']

        # encoded_inputs = {
        #     "pixel_values": transformed_image,
        #     "labels": transformed_mask
        # }

        return encoded_inputs
    

