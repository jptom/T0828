# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import dataset
import utils

max_epoches = 60
batch_size = 64
lr = 0.01

if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                            torchvision.transforms.Normalize((0.5,), (0.5,)),
                                            torchvision.transforms.RandomHorizontalFlip(),
                                            torchvision.transforms.RandomRotation((-20, 20))])
    Dataset = dataset.CAR("dataset", train=True)
    train_x, train_y = Dataset.train()
    idx = [i for i in range(Dataset.train_size)]
    IdxLoader = utils.IdxShuffler(Dataset.train_size, batch_size)
    
    resnet = torchvision.models.resnet34(num_classes=1000)
    resnet.load_state_dict(torch.load("resnet34-333f7ec4.pth"))
    model = nn.Sequential(resnet,
                          nn.Linear(1000, 500),
                          nn.ReLU(inplace=True),
                          nn.Linear(500, Dataset.num_classes))
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    for epoch in range(max_epoches):
        running_loss = 0
        
        for i, idxes in enumerate(IdxLoader, 0):
            idxes = sorted(idxes)
            inputs = torch.stack([trans(im) for im in train_x[idxes]]).to(device)
            labels = torch.from_numpy(train_y[idxes]).long().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print("epoch {}, loss: {}".format(epoch+1, running_loss/(i+1)))
        
    torch.save(model.state_dict(), "model.pth")
        
