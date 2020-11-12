# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:19:56 2020

@author: elevator-talk
"""

import torch
import torch.nn as nn
import torchvision
import dataset
import pandas as pd
import glob
import os

header = ["id", "label"]

if __name__ == "__main__":
    ids = glob.glob("dataset/testing_data/*.*")
    print(len(ids))
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                            torchvision.transforms.Normalize((0.5,), (0.5,))])
    
    train_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                            torchvision.transforms.Normalize((0.5,), (0.5,)),
                                            torchvision.transforms.RandomHorizontalFlip(),
                                            torchvision.transforms.RandomRotation((-20, 20))])
    
    Dataset = dataset.CAR("dataset", train=True)
    
    test_x,  = Dataset.test()
    resnet = torchvision.models.resnet34(num_classes=1000)
    model = nn.Sequential(resnet,
                          nn.Linear(1000, 500),
                          nn.ReLU(inplace=True),
                          nn.Linear(500, Dataset.num_classes))
    
    model.load_state_dict(torch.load("model.pth"))
    model.to(device)
    model.eval()
    
    results = []
    
    for i in range(Dataset.test_size):
        filename = os.path.splitext(os.path.basename(ids[i]))[0]
        tmp = test_x[i]
        image = trans(tmp).unsqueeze(0).to(device)
        output = model(image)
        _, result = torch.max(output, 1)
        
        label = Dataset.labeldict[result.item()]

        results.append([filename, label])    
        
        
    df = pd.DataFrame(results, columns=header)
    df.to_csv('result.csv', index=False)    
        
        
        