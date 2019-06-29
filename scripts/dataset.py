import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from imgaug import augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt
import os


class PillDataset(Dataset):
    def __init__(self, df, label_encoder, labelcol='label', augment=False):
        self.df = df
        self.label_encoder = label_encoder
        self.augment = augment
        self.aug_iaa = iaa.Sequential([
            iaa.Sometimes(0.8, iaa.GaussianBlur(sigma=(0, 1.0))),
            iaa.Affine(rotate=(-180, 180),
                       scale=(0.9, 1.0),
                       translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}),
            iaa.AddToHueAndSaturation(value=(-20, 20), per_channel=True),
            iaa.ContrastNormalization((0.8, 1.2), per_channel=True),
            iaa.Sometimes(0.8, iaa.Affine(
                shear=(-5, 5),
            ))
        ])

        self.labelcol = labelcol
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
        ])

    def __getitem__(self, index):
        img_row = self.df.iloc[index]

        img_path = img_row.images
        img = self.load_img(img_path)
        img = Image.fromarray(img)
        img = self.transform(img)

        pill_label = img_row[self.labelcol]

        data = {
            'image': img,
            'label': self.label_encoder.transform([pill_label])[0],
            'pill_label': pill_label,
            'image_name': img_path
        }

        return data

    def __len__(self):
        return len(self.df)

    def load_img(self, img_path):
        if not os.path.exists(img_path):
            print("img not found", img_path)
            return

        img = np.asarray(Image.open(img_path))

        if self.augment:
            img = self.aug_iaa.augment_image(img)

        return img

# import torchvision.utils

# def reverse_transform(inp):
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     inp = (inp * 255).astype(np.uint8)

#     return inp

# # Get a batch of training data
# batch = next(iter(dataloaders['train']))

# print(batch['image'].shape, batch['label'], batch['pill_label'])

# for i in range(20):
#     plt.figure()
#     plt.imshow(reverse_transform(batch['image'][i]))
