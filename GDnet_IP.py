##################################################################################
# Performing 3-kold cross validation using GDnet-IP model
#
# Authors: 
#   - Dongcheng Li (dongchengli287@gmail.com)
#   - Zhijun Dai (daizhijun@hunau.edu.cn)
# Date Created: 09-01-2024
# Version: 1.0
#
# Usage: python GDnet_IP_kfold.py
#
# Dependencies:
#   - Python >= 3.x
#   - Required Libraries: numpy, collections, scipy, pandas, matplotlib, math, torchvision, torch, PIL
#
#
##################################################################################

import os, random, time, prepData, defGDnet
import numpy as np
from collections import Counter

import torchvision, torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR ,MultiStepLR


def set_seed(seed): # Define the seed of random numbers
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def extract(predict, label): # Extract the predicted and observed labels
    list_p = []
    list_b = []
    list_p = [i.item() for i in predict]
    list_b = [i.item() for i in label]
    return list_p, list_b

def evaluate(y_prediction, label, num_class): # Calculate the evaluation measure (Balanced Accuracy)
    acc = []
    for cla in range(num_class):
        cla_num = 0
        inde = 0
        right = 0
        for i in label:
            if i == cla:
                cla_num += 1 
        for i in label:
            if i == cla:
                start = inde
                end = start + cla_num
                break
            else:
                inde += 1
        y = y_prediction[start:end]
        for j in y:
            if j == cla:
                right += 1
        acc.append((right / cla_num))
    return sum(acc) / num_class

def kfold3(seed, root_path, epoch_num=43, k=[1,2,3]): # Perform 3-fold cross validation similar to CPAFNet
    for fold in k:
        set_seed(seed)
        start_time = time.time()
        ba_list = []
        file_name = f'defGDnet_{random}_{seed}_{fold}.txt'
        device = torch.device("cuda:0")
        print(device)
        net = defGDnet.GDnet("V1")
        net.to(device)
        loss_entropy = nn.CrossEntropyLoss()
        optimizar = optim.Adam(net.parameters(), lr=0.0008)
        is_run = 2

        train_images_path,train_images_label,val_images_path,val_images_label = prepData.read_data(root_path,fold=fold)
        train_images_path = [path for path in train_images_path if not path.endswith('.ipynb_checkpoints')]
        train_dataset = prepData.Mydata(image_path=train_images_path, images_class=train_images_label, transform=True)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=10)

        val_images_path = [path for path in val_images_path if not path.endswith('.ipynb_checkpoints')]
        val_dataset = prepData.valMydata(image_path=val_images_path, images_class=val_images_label, transform=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=10)

        iteration_times = 0
        a = 0
        index = 0
        best_acc = 0.0
        acc = 0.0
        acc_list = []
        loss_list=[]
        c = -1
        for epoch in range(1,epoch_num+1):
            net.train()
            if epoch >= 2:
                index += 1
                if index > 3:
                    index -= 3
                    a += 1
                if a % 2 != 0:
                    optimizar.param_groups[0]["lr"] = 0.0008*0.1
                if a % 2 == 0:
                    optimizar.param_groups[0]["lr"] = 0.0008
            for step, data in enumerate(train_loader):
                img, label = data
                optimizar.zero_grad()
                output = net(img.to(device), label.to(device))
                output = output.reshape(-1, 20)
                loss = loss_entropy(output, label.to(device))
                loss.backward()
                optimizar.step()
                loss_list.append(loss.item())
                iteration_times += 1
                if iteration_times % 10 == 0:
                    print("Times of training: {}; Current loss: {}".format(iteration_times, loss.item()))
                    print("Epoch: {}".format(epoch))
            if a %2 != 0 and epoch > 9:
                acc = 0.0
                with torch.no_grad():
                    net.eval() # validation
                    list_p, list_b = [], []
                    for img, label in val_loader:
                        img = img.to(device)
                        output = net(img, label)
                        output = output.reshape(-1, 20)
                        predict = torch.max(output, dim=1)[1]  
                        list_temp_p, list_temp_b = extract(predict, label)
                        list_p = list_p + list_temp_p
                        list_b = list_b + list_temp_b
                        acc += (predict == label.to(device)).sum().item()
                acc=acc/1484
                acc_list.append(acc)
                balanced_accuracy = evaluate(list_p, list_b, num_class=20)
                print("BC is :{}".format(balanced_accuracy))
                print("AC is :{}".format(acc))
                ba_list.append(balanced_accuracy)
                print(balanced_accuracy)

        balanced_accuracy=sum(ba_list)/len(ba_list)
        fold_time = time.time() - start_time
        with open(file_name, 'w') as file:
            file.write("BC List: ")
            for bc in ba_list:
                file.write(f"{bc} ")
            file.write("Fold Times: ")
            file.write(f"{fold_time} ")
            file.write("\n")
            file.write(f"balanced_accuracy: {balanced_accuracy}\n")
        file.close()


device = torch.device("cuda:0")
print(device)
root_path = './py/data/train_data'
insect_class = [i for i in listdir(root_path) if isdir(joinPath(root_path, i))]
class_index = dict((k, v) for v, k in enumerate(insect_class))

for i in [1,2,3]:
    kfold3(i ,root_path=root_path)
