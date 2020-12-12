# Instance segmentation

## dataset
  Tiny pascal voc dataset
  - 1349 training images
  - 100 testing images
  - 20 classes
  
I use mask rcnn and it's backbone is resnet 50.  
I got 0.444 mAP for testing

# trianing and testing
There is only one .ipynb including traing and testing.  
This program work on google colab, and I recommend to use gpu mode.  
Put the dataset like this.
```bash
.
├── working directory
    ├── dataset
        ├── test_image
        ├── train_image
        ├── test.json
        ├── train.json
```
Please set your dataset path in the cell in which you set the dataset config.  
Please set your output json path in the last cell.
You can get result in about 2 hours if you use gpu mode.

