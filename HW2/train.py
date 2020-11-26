# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import Generator
from dataset import SVHN
from yolo_v1 import get_resnet34, yolo_loss
from scheduler import CustomLearningRateScheduler, lr_schedule


if __name__=="__main__":
    batch_size = 32
    div_rate = 0.8
    dataset = SVHN("dataset")
    x, y = dataset.train()
    train_x, val_x = x[:int(len(x)*div_rate)], x[int(len(x)*div_rate):]
    train_y, val_y = y[:int(len(y)*div_rate)], y[int(len(y)*div_rate):]
    train_generator = Generator(train_x, train_y, batch_size)
    val_generator = Generator(val_x, val_y, batch_size)
    model = get_resnet34()
     
    mcp_save = ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    model.compile(loss=yolo_loss, optimizer='adam')
    
    model.fit(x=train_generator, 
              steps_per_epoch=len(train_generator),
              epochs=135,
              verbose=1,
              workers=4,
              validation_data=val_generator,
              validation_steps=len(val_generator),
              callbacks=[CustomLearningRateScheduler(lr_schedule),
                         mcp_save])