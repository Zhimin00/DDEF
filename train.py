import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage as nd
from tensorflow import keras
from keras import backend as K
import pdb
from tensorflow.keras import layers
%matplotlib inline
import pylab as pl
from IPython import display
from sklearn.metrics import accuracy_score
import tensorflow_datasets as tfds
from scipy.io import loadmat as load
import torch
from utils import resnet18_DDEF, WarmUp, CustomMSE

## load original datasets
(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = tf.keras.datasets.mnist.load_data()

svhn_train = load('/kaggle/input/svhn-mat/train_32x32.mat')
svhn_test = load('/kaggle/input/svhn-mat/test_32x32.mat')
svhn_train_images = svhn_train['X']
svhn_train_labels = svhn_train['y']
svhn_test_images = svhn_test['X']
svhn_test_labels = svhn_test['y']
##svhn labels: 1-10 -> 0-9
svhn_train_labels = svhn_train_labels%10
svhn_test_labels = svhn_test_labels%10
##svhn images (H, W, C, N)->(N, H, W, C)
svhn_train_images = np.transpose(svhn_train_images, (3, 0, 1, 2))
svhn_test_images = np.transpose(svhn_test_images, (3, 0, 1, 2))

## pair two datasets to construct pseduo multimodal dataset

idx1 = torch.load('/kaggle/input/train-test-idx/train-ms-mnist-idx.pt')
idx2 = torch.load('/kaggle/input/train-test-idx/train-ms-svhn-idx.pt')
test_idx1 = torch.load('/kaggle/input/train-test-idx/test-ms-mnist-idx.pt')
test_idx2 = torch.load('/kaggle/input/train-test-idx/test-ms-svhn-idx.pt')

modal_mnist_samples = mnist_train_images[idx1]
modal_svhn_samples = svhn_train_images[idx2]
modals_labels = mnist_train_labels[idx1]

modal_mnist_tests = mnist_test_images[test_idx1]
modal_svhn_tests = svhn_test_images[test_idx2]
modals_test_labels = mnist_test_labels[test_idx1]

print(modals_labels.shape)
print(len(test_idx1))


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
mnist_train_labels_hot = dense_to_one_hot(mnist_train_labels)
mnist_test_labels_hot = dense_to_one_hot(mnist_test_labels)
svhn_train_labels_hot = dense_to_one_hot(svhn_train_labels)
svhn_test_labels_hot = dense_to_one_hot(svhn_test_labels)
modals_labels_hot = dense_to_one_hot(modals_labels)
modals_test_labels_hot = dense_to_one_hot(modals_labels)


model_DDEF = resnet18_DDEF(class_nums=10,input_shape1=(28,28,1), input_shape2=(32,32,3))
#model_DDEF.summary()
annealing_step = 10*(len(modals_labels)//64)
model_DDEF.compile(optimizer=tf.keras.optimizers.Adam(),loss=CustomMSE(), metrics=['accuracy'])
warm_up_ac = WarmUp(total_steps=annealing_step,global_step_init=0,ac=0,verbose=0)
history_DDEF=model_DDEF.fit((modal_mnist_samples,modal_svhn_samples), modals_labels_hot, batch_size=64, epochs=10,verbose=1,callbacks=[warm_up_ac])
test_preds_DDEF = model_DDEF.predict((modal_mnist_tests,modal_svhn_tests))
y_preds_DDEF = np.argmax(test_preds_DDEF, axis=1)
acc_score_DDEF = accuracy_score(y_true=modals_test_labels,y_pred=y_preds_DDEF)
print(acc_score_DDEF)