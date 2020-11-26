# -*- coding: utf-8 -*-
from utils import Generator
from dataset import SVHN
from yolo_v1 import get_resnet34, convert_result
import json

if __name__ == "__main__":
    submmit = True
    batch_size = 32
    dataset = SVHN("dataset")
    test_x, test_y = dataset.test()
    test_generator = Generator(test_x, test_y, batch_size, size=True)
    model = get_resnet34()
    results = []
    for i in range(len(test_generator)):
        batch_x, batch_y, size = test_generator[i]
        output = model.predict(batch_x)
        result = convert_result(batch_x, size)
        results.extend(result)
    
    if submmit:
       with open("submmit.json", "w") as f:
           json.dumps(results, f)