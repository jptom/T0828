# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pandas as pd
import os
import glob
import h5py

HEIGHT = 224
WIDTH = 224
CHANNELS = 3
SHAPE = (HEIGHT, WIDTH, CHANNELS)
dtype = {'id':'str', 'label':'str'}

class CAR:
    def __init__(self, dire, transform=None, train=True):
        '''
        car dataset
        '''        
        self.data(dire)
        
        
    def __del__(self):
        if self.hf:
            self.hf.close()
            
    def __getitem__(self, item):
        return self.labeldict[item]
        
    def train(self):
        return (self.train_x, self.train_y)
    
    def test(self):
        return (self.test_x, )
        
    def download(self):
        pass
    
    def read_csv(self, dire, header=True):
        df = pd.read_csv(os.path.join(dire, "training_labels.csv"), dtype=dtype)
                        
        return list(df['id']), list(df['label'])
    
    def make_label(self, modelnames):
        modelnameset = set()
        labeldict = list()
        labels = []
        
        for modelname in modelnames:
            modelnameset.add(modelname)
        
        labeldict = list(modelnameset)

        for modelname in modelnames: 
            labels.append(labeldict.index(modelname))
            
        return np.array(labels), labeldict
        
        
    def read_images(self, dire, filenames=None):
        if not os.path.isdir(dire):
            raise ValueError("%s, no such directory" % dire)
        
        if type(filenames) is list or type(filenames) is tuple:
            images = np.zeros((len(filenames), HEIGHT, WIDTH, CHANNELS))
            for i in range(len(filenames)):
                filepath = glob.glob(os.path.join(dire, filenames[i]+".*"))
                if not filepath:
                    raise ValueError("cannot find %s" % filenames[i])
                    
                image_tmp = cv2.imread(filepath[0])/255
                image_tmp = cv2.resize(image_tmp, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC).astype(int)
                images[i] = image_tmp
        
        else:
            filenames = glob.glob(os.path.join(dire, "*.*"))
            images = np.zeros((len(filenames), HEIGHT, WIDTH, CHANNELS))
            for i in range(len(filenames)):
                image_tmp = cv2.imread(filenames[i])/255
                image_tmp = cv2.resize(image_tmp, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC).astype(int)
                images[i] = image_tmp
                
        return images
            
    def to_h5(self, dire, train_x, train_y, test_x, labeldict):
        with h5py.File(os.path.join(dire, "data.h5"), 'w') as hf:
            if type(train_x) is np.ndarray:
                hf.create_dataset(name='train_x',
                                  data=train_x,
                                  dtype=np.uint8,
                                  shape=train_x.shape,
                                  compression="gzip",
                                  compression_opts=9)
            if type(train_y) is np.ndarray:
                hf.create_dataset(name='train_y',
                                  data=train_y,
                                  shape=train_y.shape,
                                  dtype=np.int,
                                  compression="gzip",
                                  compression_opts=9)
            if type(test_x) is np.ndarray:
                hf.create_dataset(name='test_x',
                                  data=test_x,
                                  shape=test_x.shape,
                                  dtype=np.uint8,
                                  compression="gzip",
                                  compression_opts=9)
            labeldict_h5 = hf.create_dataset(name='labeldict',
                                             shape=(len(labeldict), ),
                                             dtype=h5py.special_dtype(vlen=str),
                                             compression="gzip",
                                             compression_opts=9)
            
            for i in range(len(labeldict)):
                labeldict_h5[i] = labeldict[i]
    
    def load_h5(self, dire):# not complete
        self.hf = h5py.File(os.path.join(dire, "data.h5"), 'r') 
        self.labeldict = self.hf['labeldict']
        self.train_x = self.hf['train_x']
        self.train_y = self.hf['train_y']
        self.test_x = self.hf['test_x']
        self.train_size = self.train_x.shape[0]
        self.test_size = self.test_x.shape[0]
        self.num_classes = len(self.labeldict)
        
        
    
    def to_npz(self, dire, train_x, train_y, test_x,):
        np.savez_compressed(os.path.join(dire, "data.npz"), train_x=train_x, train_y=train_y, test_x=test_x)  

    def load_npz(self, dire) -> dict:
        data = np.load(os.path.join(dire, "data.npz"))
        return data["train_x"], data["train_y"], data["test_x"]
        
    def data(self, dire):
        if not os.path.isfile(os.path.join(dire, "data.h5")):
            if not os.path.isdir(os.path.join(dire, "training_data")) \
                or not os.path.isdir(os.path.join(dire, "testing_data")) \
                or not os.path.isfile(os.path.join(dire, "training_labels")):
                self.download()
            
            
            filenames, modelnames = self.read_csv(dire)
            train_images = self.read_images(os.path.join(dire, "training_data"), filenames)
            train_labels, labeldict = self.make_label(modelnames)
            test_images = self.read_images(os.path.join(dire, "testing_data"))
            
            self.to_h5(dire, train_images, train_labels, test_images, labeldict)
        
        self.load_h5(dire)
    
    
    def label_dict(self, dire):
        if not os.path.isfile(os.path.join(dire, "labeldict.pickle")):
            if not os.path.isdir(os.path.join(dire, "training_data")) \
                or not os.path.isdir(os.path.join(dire, "testing_data")) \
                or not os.path.isfile(os.path.join(dire, "training_labels")):
                self.download()
            
            filenames, modelnames = self.read_csv(dire)
            _, labeldict = self.make_label(modelnames)
            self.pickle_dump(dire, labeldict)
        
        return self.pickle_load(dire)
        
    def open_database(self, dire):
        self.f = h5py.File(os.path.join(dire, "data.h5"))
        
    def mean_std(self, train_x, test_x):
        mean = np.zeros(CHANNELS)
        std = np.zeros(CHANNELS)
        
        for i in range(len(train_x)):
            mean += train_x.mean(axis=(0, 1))
            std += train_x.std(axis=(0, 1))
            
        for i in range(len(test_x)):
            mean += test_x.mean(axis=(0, 1))
            std += test_x.std(axis=(0, 1))
            
        mean /= len(train_x) + len(test_x)
        std /= len(train_x) + len(test_x)
        
        return mean, std
    

        
        
if __name__ == "__main__":
    Dataset = CAR("")
    
    