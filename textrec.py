import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
from tensorflow.examples.tutorials.mnist import input_data

## setting the config for neural network
##followed tutorial form https://www.youtube.com/watch?v=HMcx-zY8JSg&t=605s
print("\nSetting NN Settings:\n \nLayer One:\n \tfilter size 15\n\t16 filters\n")
layer1_filterSize = 5
layer1_numFilters = 16
print("Layer 2:\n\t Filter size 5\n\t 36 filters\n")
layer2_filterSize = 5
layer2_numFilters = 36
print("Number of neurons in fully connected layer: 128")
fc_size = 128

## Getting the mnist dataset
print("Beginning Retrieval of MNIST Dataset...\n")

data = input_data.read_data_sets('data/MNIST', one_hot = True)

print("Data retrieval complete.\n")

## setting the data dimensions
print("Setting data dimensions...\n")
imgSize = 28
imgSizeFlat = imgSize * imgSize
imgShape = (imgSize, imgSize)
numChannels = 1

## 10 classification labels
numClasses = 10
testDataClasses = np.argmax(data.test.labels, axis = 1)
print("Data dimensions set.")
## Create image plots with matplot lib
def plotImages(images, CLS_true, CLS_pred = None):
    assert len(images) == len(CLS_true) == 9

    fig, axes = plt.subplots(3,3)
    fig.subplots_adjust(hspace = 0.3, wspace = 0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(imgShape), cmap='binary')
        if CLS_pred is None:
            xlabel = "True Label {0}".format(CLS_true[i])
        else:
            xlabel = "True Label {0}, Predicted Label: {2}".format(CLS_true[i], CLS_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

testImage = data.test.images[0:9]
CLS_true = testDataClasses[0:9]
plotImages(testImage, CLS_true)

def newWeights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.05))

def newBiases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def newConvLayer(input, numInputChannels, filterSize, numFilters, usePooling = True):
    shape = [filterSize, filterSize, numInputChannels, numFilters]
    weights = newWeights(shape = shape)
    biases = newBiases(length = numFilters)

    layer = tf.nn.conv2d(input = input, filter = weights, strides = [1,1,1,1], padding = 'SAME')
    layer += biases

    if usePooling:
        layer = tf.nn.max_pool(value = layer, ksize = [1,2,2,1], strides =[1,2,2,1], padding = 'SAME')

    layer = tf.nn.relu(layer)
    return layer, weights

def flattenLayer(layer):
    layerShape = layer.get_shape()
    numFeatures = np.array(layerShape[1:4], dtype = int).prod()
    layerFlat = tf.reshape(layer, [-1, numFeatures])
    return layerFlat, numFeatures

def newFCLayer(input, numInputs, numOutputs, useRelu = True):
    weights = newWeights(shape=[numInputs, numOutputs])
    biases = newBiases(length = numOutputs)
    layer = tf.matmul(input, weights) + biases

    if useRelu:
        layer = tf.nn.relu(layer)

    return layer


##placeholder

x = tf.placeholder(tf.float32, shape = [None, imgSizeFlat], name = 'x')

##placeholder as image

xImage = tf.reshape(x, [-1, imgSize, imgSize, numChannels])

yTrue = tf.placeholder(tf.float32, shape = [None, 10], name ='yTrue')
yTrueCls = tf.argmax(yTrue, dimension = 1)

layerConv1, weightsConv1 = newConvLayer(input = xImage, numInputChannels = numChannels,filterSize = layer1_filterSize, numFilters = layer1_numFilters, usePooling = True)

layerConv2, weightsConv2 = newConvLayer(input = layerConv1, numInputChannels = layer1_numFilters,filterSize = layer2_filterSize, numFilters = layer2_numFilters, usePooling = True)

layerFlat, numFeatures = flattenLayer(layerConv2)

print(numFeatures)


