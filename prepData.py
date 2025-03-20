import matplotlib.pyplot as plt
import numpy as np
import math
import csv

from os import listdir
from os.path import isdir
import os.path.join as joinPath

import torchvision.models
import torchvision.transforms as tvTrans
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn as nn


class Mydata(Dataset): # Define the structure of training set
    def __init__(self, image_path: list, images_class: list, transform):
        self.images_path = image_path
        self.images_class = images_class
        self.transform = transform
    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        img = Image.open(self.images_path[index])
        label = self.images_class[index]
        if self.transform == True:
            transformer = tvTrans.Compose([tvTrans.Resize(224), tvTrans.RandomHorizontalFlip(), tvTrans.ToTensor(), tvTrans.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
            img = transformer(img)
        return img, label


class valMydata(Dataset): # Define the structure of validation set
    def __init__(self, image_path: list, images_class: list, transform):
        self.images_path = image_path
        self.images_class = images_class
        self.transform = transform
    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        img = Image.open(self.images_path[index])
        label = self.images_class[index]
        if self.transform == True:
            transformer = tvTrans.Compose([tvTrans.Resize(224),tvTrans.ToTensor(),tvTrans.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
            img = transformer(img)
        return img, label

# # train_images_path, train_images_label, val_images_path, val_images_label = read_data(root_path,1)
# train_dataset = Mydata(image_path=train_images_path, images_class=train_images_label, transform=True)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
# val_dataset = Mydata(image_path=val_images_path, images_class=val_images_label, transform=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)


def read_data(root:str, fold:int, da_root='./py/data//train_data_da'):
    insect_class = [i for i in listdir(root) if isdir(joinPath(root,i))]
    class_index = dict((k,v) for v,k in enumerate(insect_class))
    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    if fold == 1: # Prepare data for the first fold validation
        for cla in insect_class:
            insect_path = joinPath(root,cla,"D2")
            insect_path2 = joinPath(root,cla,"D3")

            images = [joinPath(root,cla,"D2",i) for i in listdir(insect_path)]
            images2 = [joinPath(root,cla,"D3",i) for i in listdir(insect_path2)]
            images = images + images2 # Images used for training
            images_class = class_index[cla]

            val_path = joinPath(root,cla,"D1") # Path of validation set
            val_images = [joinPath(root,cla,"D1",i) for i in listdir(val_path)]
            val_class = class_index[cla]

            insect_da_path = joinPath(da_root, cla, "D2") # Path for the augmented images
            insect_da_path2 = joinPath(da_root, cla, "D3")
            da_images_temp = [joinPath(da_root, cla, "D2", i) for i in listdir(insect_da_path)]
            da_images_temp2 = [joinPath(da_root, cla, "D3", i) for i in listdir(insect_da_path2)]
            da_images = da_images_temp2 + da_images_temp # Augmented images used for training
            da_images_class = class_index[cla]

            for img_path in images:
                train_images_path.append(img_path)
                train_images_label.append(images_class)
            for img_path in da_images:
                train_images_path.append(img_path)
                train_images_label.append(da_images_class)
            for img_path in val_images:
                val_images_path.append(img_path)
                val_images_label.append(val_class)

    if fold==2: # Prepare data for the second fold validation
        for cla in insect_class:
            insect_path = joinPath(root, cla, "D1")
            insect_path2 = joinPath(root, cla, "D3")

            images = [joinPath(root, cla, "D1", i) for i in listdir(insect_path)]
            images2 = [joinPath(root, cla, "D3", i) for i in listdir(insect_path2)]
            images = images + images2
            images_class = class_index[cla]

            val_path = joinPath(root, cla, "D2")
            val_images = [joinPath(root, cla, "D2", i) for i in listdir(val_path)]
            val_class = class_index[cla]

            insect_da_path = joinPath(da_root, cla, "D1")
            insect_da_path2 = joinPath(da_root, cla, "D3")
            da_images_temp = [joinPath(da_root, cla, "D1", i) for i in listdir(insect_da_path)]
            da_images_temp2 = [joinPath(da_root, cla, "D3", i) for i in listdir(insect_da_path2)]
            da_images = da_images_temp2 + da_images_temp
            da_images_class = class_index[cla]

            for img_path in images:
                train_images_path.append(img_path)
                train_images_label.append(images_class)
            for img_path in da_images:
                train_images_path.append(img_path)
                train_images_label.append(da_images_class)
            for img_path in val_images:
                val_images_path.append(img_path)
                val_images_label.append(val_class)

    if fold==3: # Prepare data for the third fold validation
        for cla in insect_class:
            insect_path = joinPath(root, cla, "D2")
            insect_path2 = joinPath(root, cla, "D1")

            images = [joinPath(root, cla, "D2", i) for i in listdir(insect_path)]
            images2 = [joinPath(root, cla, "D1", i) for i in listdir(insect_path2)]
            images = images + images2
            images_class = class_index[cla]

            val_path = joinPath(root, cla, "D3")
            val_images = [joinPath(root, cla, "D3", i) for i in listdir(val_path)]
            val_class = class_index[cla]

            insect_da_path = joinPath(da_root, cla, "D2")
            insect_da_path2 = joinPath(da_root, cla, "D1")
            da_images_temp = [joinPath(da_root, cla, "D2", i) for i in listdir(insect_da_path)]
            da_images_temp2 = [joinPath(da_root, cla, "D1", i) for i in listdir(insect_da_path2)]
            da_images = da_images_temp2 + da_images_temp
            da_images_class = class_index[cla]

            for img_path in images:
                train_images_path.append(img_path)
                train_images_label.append(images_class)
            for img_path in da_images:
                train_images_path.append(img_path)
                train_images_label.append(da_images_class)
            for img_path in val_images:
                val_images_path.append(img_path)
                val_images_label.append(val_class)

    return train_images_path,train_images_label,val_images_path,val_images_label


def a(list):
    return sum(list)/len(list)

def read_data_for_generator(root:str, da_root=".\py\data\\train_data_da"):
    insect_class = [i for i in listdir(root) if isdir(joinPath(root,i))]
    class_index = dict((k,v) for v,k in enumerate(insect_class))
    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    for cla in insect_class:
        insect_path = joinPath(root,cla,"D2")
        insect_path2 = joinPath(root,cla,"D3")
        insect_path3 = joinPath(root,cla,"D1")

        images = [joinPath(root,cla,"D2",i) for i in listdir(insect_path)]
        images2 = [joinPath(root,cla,"D3",i) for i in listdir(insect_path2)]
        images3 = [joinPath(root,cla,"D1",i) for i in listdir(insect_path3)]
        images = images + images2 + images3
        images_class = class_index[cla]

        val_path = joinPath(root,cla,"D1")
        val_images = [joinPath(root,cla,"D1",i) for i in listdir(val_path)]
        val_class = class_index[cla]

        insect_da_path = joinPath(da_root, cla, "D2")
        insect_da_path2 = joinPath(da_root, cla, "D3")
        insect_da_path3 = joinPath(da_root, cla, "D1")

        da_images_temp = [joinPath(da_root, cla, "D2", i) for i in listdir(insect_da_path)]
        da_images_temp2 = [joinPath(da_root, cla, "D3", i) for i in listdir(insect_da_path2)]
        da_images_temp3 = [joinPath(da_root, cla, "D1", i) for i in listdir(insect_da_path3)]
        da_images = da_images_temp2 + da_images_temp2 + da_images_temp3
        da_images_class = class_index[cla]

        for img_path in images:
            train_images_path.append(img_path)
            train_images_label.append(images_class)
        for img_path in da_images:
            train_images_path.append(img_path)
            train_images_label.append(da_images_class)
        for img_path in val_images:
            val_images_path.append(img_path)
            val_images_label.append(val_class)
    return train_images_path, train_images_label, val_images_path, val_images_label

def read_data2(root, test): # Prepare data for independent test
    insect_class = [i for i in listdir(root) if isdir(joinPath(root,i))]
    class_index = dict((k,v) for v,k in enumerate(insect_class))
    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    val_images_train_path = []
    val_labels_train_path = []
    every_class_num = []
    for cla in insect_class:
        insect_path = joinPath(root, cla, "D1")
        insect_path2 = joinPath(root,cla,"D2")
        insect_path3 = joinPath(root,cla,"D3")

        images = [joinPath(root,cla,"D1",i) for i in listdir(insect_path)]
        images2 = [joinPath(root,cla,"D2",i) for i in listdir(insect_path2)]
        images3 = [joinPath(root,cla,"D3",i) for i in listdir(insect_path3)]
        images = images + images2 + images3
        images_class = class_index[cla]

        val_path = joinPath(test,cla) # Path prepared for test set
        val_images = [joinPath(test,cla,i) for i in listdir(val_path)]
        val_class = class_index[cla]
        val_train_class = class_index[cla]
        # val_path_train = joinPath(test_train,cla)
        # val_images_train = [joinPath(test_train,cla,i) for i in listdir(val_path_train)]
        
        insect_da_path = joinPath(da_root, cla, "D1")
        insect_da_path2= joinPath(da_root, cla, "D2")
        insect_da_path3 = joinPath(da_root, cla, "D3")
        da_images_temp = [joinPath(da_root, cla, "D1", i) for i in listdir(insect_da_path)]
        da_images_temp2 = [joinPath(da_root, cla, "D2", i) for i in listdir(insect_da_path2)]
        da_images_temp3 = [joinPath(da_root, cla, "D3", i) for i in listdir(insect_da_path3)]
        da_images = da_images_temp + da_images_temp2 + da_images_temp3
        da_images_class = class_index[cla]

        for img_path in images:
            train_images_path.append(img_path)
            train_images_label.append(images_class)
        for img_path in da_images:
            train_images_path.append(img_path)
            train_images_label.append(da_images_class)
        for img_path in val_images:
            val_images_path.append(img_path)
            val_images_label.append(val_class)
        # for img_path in val_images_train:
        #     val_images_train_path.append(img_path)
        #     val_labels_train_path.append(val_class)
    return train_images_path, train_images_label, val_images_path, val_images_label

def read_data3(root, test_train):
    insect_class = [i for i in listdir(root) if isdir(joinPath(root,i))]
    class_index = dict((k,v) for v,k in enumerate(insect_class))
    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    val_images_train_path = []
    val_labels_train_path = []
    every_class_num = []
    for cla in insect_class:
        val_path = joinPath(test_train, cla)
        val_images = [joinPath(test_train,cla,i) for i in listdir(val_path)]

        val_class = class_index[cla]
        val_train_class = class_index[cla]
        for img_path in val_images:
            val_images_path.append(img_path)
            val_images_label.append(val_class)

    return val_images_path, val_images_label


# --- Specify the file path relating to the insect data ---
da_root = "./py/data//train_data_da" # Augmented images used for training (excluding images for testing)
root_path = "./py/data//train_data" # Original images used for training (excluding images for testing)
test_path = "./py/data//test" # Images sampled from the original dataset set to be used for testing



