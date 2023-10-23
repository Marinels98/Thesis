#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import numpy as np
import torch
import os
from torchvision import datasets
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from typing import List, Callable, Tuple, Generator
from torch.utils.data import ConcatDataset

data_transform = transforms.Compose([
    # Modifica le dimensioni delle immagini come necessario
    transforms.ToTensor(),
])







class CMNIST(Dataset):
    def __init__(self, data_dir="./data/cmnist", env="train",bias_amount=0.95, transform=data_transform):
        self.data_dir = data_dir
        self.transform = transform
        self.env=env
        self.bias_amount=bias_amount

        self.bias_folder_dict = {
            0.95: "5pct",
            0.98: "2pct",
            0.99: "1pct",
            0.995: "0.5pct",

        }
        if self.env == "train":
            self.samples, self.class_labels, self.bias_labels = self.load_train_samples()

        if self.env == "val":
            self.samples, self.class_labels, self.bias_labels = self.load_val_samples()

        if self.env == "test":
            self.samples, self.class_labels, self.bias_labels = self.load_test_samples()

    def load_train_samples(self):
        samples_path = []
        class_labels=[]
        bias_labels=[]
        bias_folder=self.bias_folder_dict[self.bias_amount]
        for class_folder in sorted(os.listdir(os.path.join(self.data_dir,bias_folder, "align"))):
            for filename in sorted(os.listdir(os.path.join(self.data_dir,bias_folder, "align",class_folder))):
                samples_path.append(os.path.join(self.data_dir,bias_folder, "align",class_folder,filename))
                class_labels.append(self.assign_class_label(filename))
                bias_labels.append(self.assign_bias_label(filename))
        
        for class_folder in sorted(os.listdir(os.path.join(self.data_dir,bias_folder, "conflict"))):
            for filename in sorted(os.listdir(os.path.join(self.data_dir,bias_folder, "conflict",class_folder))):
                samples_path.append(os.path.join(self.data_dir,bias_folder, "conflict",class_folder,filename))
                class_labels.append(self.assign_class_label(filename))
                bias_labels.append(self.assign_bias_label(filename))

        return (
            np.array(samples_path), 
            np.array(class_labels), 
            np.array(bias_labels)
        ) 
    
    def load_val_samples(self):
        samples_path = []
        class_labels=[]
        bias_labels=[]

        bias_folder=self.bias_folder_dict[self.bias_amount]
        for filename in sorted(os.listdir(os.path.join(self.data_dir,bias_folder,"valid"))):
            samples_path.append(os.path.join(self.data_dir,bias_folder, "valid",filename))
            class_labels.append(self.assign_class_label(filename))
            bias_labels.append(self.assign_bias_label(filename))

        return (
            np.array(samples_path), 
            np.array(class_labels), 
            np.array(bias_labels)
        ) 
    
    def load_test_samples(self):
        samples_path = []
        class_labels=[]
        bias_labels=[]

        for class_folder in sorted(os.listdir(os.path.join(self.data_dir,"test"))):
            for filename in  sorted(os.listdir(os.path.join(self.data_dir,"test",class_folder))):
                samples_path.append(os.path.join(self.data_dir,"test",class_folder,filename))
                class_labels.append(self.assign_class_label(filename))
                bias_labels.append(self.assign_bias_label(filename))

        return (
            np.array(samples_path), 
            np.array(class_labels), 
            np.array(bias_labels)
        ) 


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        class_label=self.class_labels[idx]
        bias_label=self.bias_labels[idx]

        image = self.transform(Image.open(file_path))   #senza self.transofrm per vedere le immagini 

        
        return image, class_label, bias_label

    def assign_bias_label(self, filename):
        no_extension=filename.split('.')[0]
        _, y, z = no_extension.split('_')
        y, z = int(y), int(z)
        if y == z:
            return 1
        return -1
    
    def assign_class_label(self, filename):
        no_extension=filename.split('.')[0]
        _, y, _ = no_extension.split('_')
        return int(y)
     

     

if __name__ == "__main__":


    train_set=CMNIST(env="train",bias_amount=0.95)
    val_set=CMNIST(env="val",bias_amount=0.95)
    test_set=CMNIST(env="test",bias_amount=0.95)


    #group and display colorized images of the same digit together.
    plt.figure()
    for i in range(0, 55000, 500):
        train_image, l, bl = train_set[i]
        print(train_set.samples[i])
        print("class ", l)
        print("bias ", bl)
        plt.imshow(train_image)

        plt.show()

    for i in range(0, 300, 50):
        val_image, l, bl = val_set[i]
        print(val_set.samples[i])
        print("class ", l)
        print("bias ", bl)
        plt.imshow(val_image)

        plt.show()

    for i in range(0, 4000, 100):
        test_image, l, bl = test_set[i]
        print(test_set.samples[i])
        print("class ", l)
        print("bias ", bl)
        plt.imshow(test_image)

        plt.show()