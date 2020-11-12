# -*- coding: utf-8 -*-
import random


class IdxShuffler:
    def __init__(self, n:int, batch_size:int):
        self.n = n
        self.idxes = [i for i in range(self.n)]
        self.batch_size = batch_size
        self.i = 0
        
    def __iter__(self):
        self.i = 0
        random.shuffle(self.idxes)
        return self
    
    def __next__(self):
        if self.i+self.batch_size <= self.n:
            ret = self.idxes[self.i:self.i+self.batch_size]
            self.i += self.batch_size
            return ret
        
        else:
            raise StopIteration
            
            
if __name__ == "__main__":
    pass