# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import cv2 as cv

C = 10
class_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
HEIGHT = 448
WIDTH = 448
CHANNELS = 3
SHAPE = (HEIGHT, WIDTH, CHANNELS)
B = 2
GRID = 14

def read(image:pd.DataFrame, label:pd.DataFrame):
    ret_image = cv.resize(image['img'], (HEIGHT, WIDTH))
    ret_image = ret_image/255.0
    ret_label = np.zeros([GRID, GRID, B*5 + C])

    for i in range(len(label['class'])):
        x = label['x'][i]/image['width']
        y = label['y'][i]/image['height']
        w = label['width'][i]/image['width']
        h = label['height'][i]/image['height']
        loc = [GRID*x, GRID*y]
        loc_i = int(loc[1])
        loc_j = int(loc[0])
        x = loc[0] - loc_j
        y = loc[1] - loc_i
       
        if ret_label[loc_i, loc_j, 14] == 0:
            ret_label[loc_i, loc_j, label['class']] = 1
            ret_label[loc_i, loc_j, 10:14] = [x, y, w, h]
            ret_label[loc_i, loc_j, 14] = 1
        else:
            #print("conflict: ", label["img_name"])
            pass

            
    return ret_image, ret_label

def convert_result(output, size:pd.DataFrame, threshold=0.4):
    result = []
    for batch in range(len(size)):
        score = []
        bbox = []
        label = []
        for i in range(GRID):
            for j in range(GRID):
                for b in range(B): 
                    if output[batch, i, j, C+b] > threshold:
                        s = output[batch, i, j, C+b]
                        l = int(keras.backend.argmax(output[batch, i, j, :C]))
                        x, y, w, h = output[batch, i, j, C+B+4*b:C+B+4*(b+1)]
                        W = size.iloc[batch]['width']
                        H = size.iloc[batch]['height']
                        x = (x + j)/GRID * W
                        y = (y + i)/GRID * H
                        w = w*W
                        h = h*H
                        top = max(keras.backend.round(y - h/2), 0)
                        left = max(keras.backend.round(x - w/2), 0)
                        bottom = min(keras.backend.round(y + h/2), H)
                        right = min(keras.backend.round(x + w/2), W)
                        score.append(s)
                        label.append(l)
                        bbox.append((top, left, bottom, right))
        result.append({"bbox":bbox, "score":score, "label":label})
    return result
            
class ResidualUnit_v1(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides,
                                padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=1,
                                padding="same", use_bias=False),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides,
                                    padding="same", use_bias=False),
                keras.layers.BatchNormalization()]
            
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'activation': self.activation
            })
        return config
            
    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)  
    
class YOLO_OUTPUT_Layer(keras.layers.Layer):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'target_shape': self.target_shape
            })
        return config
    
    def call(self, inputs):
        softmax = keras.activations.get('softmax')
        sigmoid = keras.activations.get('sigmoid')
        
        c = softmax(inputs[:, :, :, :C])
        o = sigmoid(inputs[:, :, :, C:C+B])
        b = sigmoid(inputs[:, :, :, C+B:])
        
        outputs = keras.backend.concatenate([c, b, o])
        return outputs
    
def get_resnet34():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=SHAPE,
                                  padding="same", use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))
    prev_filters = 64
    for filters in [64]*3 + [128]*4 + [256]*6 + [512]*3:
        strides = 1 if filters == prev_filters else 2
        model.add(ResidualUnit_v1(filters, strides=strides))
        prev_filters = filters
    model.add(keras.layers.Conv2D(B*5+C, 1, padding="valid", use_bias=False))
    model.add(YOLO_OUTPUT_Layer(target_shape=(GRID, GRID, B*5+C)))
    return model

def xywh2minmax(xy, wh):
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2
    
    return xy_min, xy_max

def iou(pred_mins, pred_maxes, true_mins, true_maxes):
    intersect_mins = keras.backend.maximum(pred_mins, true_mins)
    intersect_maxes = keras.backend.minimum(pred_maxes, true_maxes)
    intersect_wh = keras.backend.maximum(intersect_maxes - intersect_mins, 0)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    pred_wh = pred_maxes - pred_mins
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]
    
    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas
    
    return iou_scores

def yolo_head(feats):
    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = keras.backend.shape(feats)[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = keras.backend.arange(0, stop=conv_dims[0])
    conv_width_index = keras.backend.arange(0, stop=conv_dims[1])
    conv_height_index = keras.backend.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = keras.backend.tile(
        keras.backend.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = keras.backend.flatten(keras.backend.transpose(conv_width_index))
    conv_index = keras.backend.transpose(keras.backend.stack([conv_height_index, conv_width_index]))
    conv_index = keras.backend.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = keras.backend.cast(conv_index, keras.backend.dtype(feats))

    conv_dims = keras.backend.cast(keras.backend.reshape(conv_dims, [1, 1, 1, 1, 2]), keras.backend.dtype(feats))

    box_xy = (feats[..., :2] + conv_index) / conv_dims * 448
    box_wh = feats[..., 2:4] * 448

    return box_xy, box_wh


def yolo_loss(y_true, y_pred):
    label_class = y_true[..., :C]  # ? * 7 * 7 * C
    label_box = y_true[..., C:C+4]  # ? * 7 * 7 * 4
    response_mask = y_true[..., C+4]  # ? * 7 * 7
    response_mask = keras.backend.expand_dims(response_mask)  # ? * 7 * 7 * 1

    predict_class = y_pred[..., :C]  # ? * 7 * 7 * C
    predict_trust = y_pred[..., C:C+2]  # ? * 7 * 7 * 2
    predict_box = y_pred[..., C+2:]  # ? * 7 * 7 * 8

    _label_box = keras.backend.reshape(label_box, [-1, GRID, GRID, 1, 4])
    _predict_box = keras.backend.reshape(predict_box, [-1, GRID, GRID, 2, 4])
    label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    label_xy = keras.backend.expand_dims(label_xy, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_wh = keras.backend.expand_dims(label_wh, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_xy_min, label_xy_max = xywh2minmax(label_xy, label_wh)  # ? * 7 * 7 * 1 * 1 * 2, ? * 7 * 7 * 1 * 1 * 2
    
    predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2
    predict_xy = keras.backend.expand_dims(predict_xy, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_wh = keras.backend.expand_dims(predict_wh, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh)  # ? * 7 * 7 * 2 * 1 * 2, ? * 7 * 7 * 2 * 1 * 2

    iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # ? * 7 * 7 * 2 * 1
    best_ious = keras.backend.max(iou_scores, axis=4)  # ? * 7 * 7 * 2
    best_box = keras.backend.max(best_ious, axis=3, keepdims=True)  # ? * 7 * 7 * 1
    box_mask = keras.backend.cast(best_ious >= best_box, keras.backend.dtype(best_ious))  # ? * 7 * 7 * 2
    
    no_object_loss = 0.5 * (1 - box_mask * response_mask) * keras.backend.square(0 - predict_trust)
    object_loss = box_mask * response_mask * keras.backend.square(1 - predict_trust)
    confidence_loss = no_object_loss + object_loss
    confidence_loss = keras.backend.sum(confidence_loss)

    class_loss = response_mask * keras.backend.square(label_class - predict_class)
    class_loss = keras.backend.sum(class_loss)

    _label_box = keras.backend.reshape(label_box, [-1, GRID, GRID, 1, 4])
    _predict_box = keras.backend.reshape(predict_box, [-1, GRID, GRID, 2, 4])

    label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2

    box_mask = keras.backend.expand_dims(box_mask)
    response_mask = keras.backend.expand_dims(response_mask)

    box_loss = 5 * box_mask * response_mask * keras.backend.square((label_xy - predict_xy) / 448)
    box_loss += 5 * box_mask * response_mask * keras.backend.square((keras.backend.sqrt(label_wh) - keras.backend.sqrt(predict_wh)) / 448)
    box_loss = keras.backend.sum(box_loss)

    loss = confidence_loss + class_loss + box_loss

    return loss

if __name__ == "__main__":
    # check model 
    resnet = get_resnet34()
    resnet.summary()
    resnet.get_config()
    
    # check loss fucntion
    a = keras.backend.random_normal((64, 14, 14, 20))
    b = keras.backend.random_normal((64, 14, 14, 20))
    loss = yolo_loss(a, b)
    
    # 
    layer = YOLO_OUTPUT_Layer((14, 14, 20))
    layer.get_config()