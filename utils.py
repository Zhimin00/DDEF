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



def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def KL(alpha):
    beta=tf.constant(np.ones((1,k)),dtype=tf.float32)
    S_alpha = tf.reduce_sum(alpha,axis=1,keepdims=True)
    S_beta = tf.reduce_sum(beta,axis=1,keepdims=True)
    lnB = tf.math.lgamma(S_alpha) - tf.reduce_sum(tf.math.lgamma(alpha),axis=1,keepdims=True)
    lnB_uni = tf.reduce_sum(tf.math.lgamma(beta),axis=1,keepdims=True) - tf.math.lgamma(S_beta)
    
    dg0 = tf.math.digamma(S_alpha)
    dg1 = tf.math.digamma(alpha)
    
    kl = tf.reduce_sum((alpha - beta)*(dg1-dg0),axis=1,keepdims=True) + lnB + lnB_uni
    print(kl)
    return kl

class CustomMSE(keras.losses.Loss):
    def __init__(self, annealing_coef=tf.Variable(initial_value=0.0,name='annealing_coef', trainable=False), name="custom_mse"):
      super().__init__(name=name)
      self.annealing_coef = annealing_coef
      self.A = 0
      self.B = 0
      self.C = 0
      #print('Inital:', self.annealing_coef)
    def call(self, y_true, y_pred):
      y_pred = tf.cast(y_pred, tf.float32)
      y_true = tf.cast(y_true, tf.float32)
      S = tf.reduce_sum(y_pred, axis=1, keepdims=True) 
      E = y_pred - 1
      m = y_pred / S
      #print('y_true:',y_true)
      A = tf.reduce_sum((y_true-m)**2, axis=1, keepdims=True) 
      B = tf.reduce_sum(y_pred*(S-y_pred)/(S*S*(S+1)), axis=1, keepdims=True)       
      alp = E*(1-y_true) + 1 
      #print('Step:',self.annealing_coef)
      C =  self.annealing_coef * KL(alp)
      self.A = A
      self.B = B
      self.C = C
      print(A,B,C)
      return tf.reduce_mean((A + B) + C)
class WarmUp(keras.callbacks.Callback):

    def __init__(self,total_steps,global_step_init=0,ac=0,
                 verbose=1):
        super(WarmUp, self).__init__()
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.ac = ac
        self.verbose = verbose
        #learning_rates
        self.acs = []
	#update global_step，and record corresponding annealing coefficient
    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        ac = K.get_value(self.model.loss.annealing_coef)
        self.acs.append(ac)
	#update annealing coefficient
    def on_batch_begin(self, batch, logs=None):
        ac = tf.minimum(1.0,tf.cast(self.global_step/self.total_steps,tf.float32)).numpy()
        #print('ac:',ac)
        K.set_value(self.model.loss.annealing_coef, ac)
        #print('after set',self.model.loss.annealing_coef)
        if self.verbose > 0:
          print('\nBatch %05d: setting annealing_coef '
                  'rate to %s.' % (self.global_step + 1, ac))
def conv2d_bn(inpt, filters=64, kernel_size=(3,3), strides=1, padding='same'):
    '''卷积、归一化和relu三合一'''
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(inpt)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def basic_bottle(inpt, filters=64, kernel_size=(3,3), strides=1, padding='same', if_baisc=False):
    '''18中的4个basic_bottle'''
    x = conv2d_bn(inpt, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
    x = conv2d_bn(x, filters=filters)
    if if_baisc==True:
        temp = conv2d_bn(inpt, filters=filters, kernel_size=(1,1), strides=2, padding='same')
        outt = layers.add([x, temp])
    else:
        outt = layers.add([x, inpt])
    return outt

def resnet18(class_nums,input_shape=(28,28,1)):
    '''main model'''
    inpt = layers.Input(shape=input_shape)
    #layer 1
    x = conv2d_bn(inpt, filters=64, kernel_size=(7,7), strides=2, padding='valid')
    x = layers.MaxPool2D(pool_size=(3,3), strides=2)(x)
    #layer 2
    x = basic_bottle(x, filters=64, kernel_size=(3,3), strides=1, padding='same', if_baisc=False)
    x = basic_bottle(x, filters=64, kernel_size=(3,3), strides=1, padding='same', if_baisc=False)
    #layer 3
    x = basic_bottle(x, filters=128, kernel_size=(3, 3), strides=2, padding='same', if_baisc=True)
    x = basic_bottle(x, filters=128, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    # layer 4
    x = basic_bottle(x, filters=256, kernel_size=(3, 3), strides=2, padding='same', if_baisc=True)
    x = basic_bottle(x, filters=256, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    # layer 5
    x = basic_bottle(x, filters=512, kernel_size=(3, 3), strides=2, padding='same', if_baisc=True)
    x = basic_bottle(x, filters=512, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    #GlobalAveragePool
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(class_nums, activation='softmax')(x)
    model = tf.keras.Model(inputs=inpt, outputs=x)
    return model

def resnet18_DDEF(class_nums,input_shape1=(28,28,1), input_shape2=(32,32,3)):
    '''main model'''
    inpt1 = layers.Input(shape=input_shape1)
    inpt2 = layers.Input(shape=input_shape2)
    ############ For the first modality, construct your DNN
    #layer 1
    x1 = conv2d_bn(inpt1, filters=64, kernel_size=(7,7), strides=2, padding='valid')
    x1 = layers.MaxPool2D(pool_size=(3,3), strides=2)(x1)
    #layer 2
    x1 = basic_bottle(x1, filters=64, kernel_size=(3,3), strides=1, padding='same', if_baisc=False)
    x1 = basic_bottle(x1, filters=64, kernel_size=(3,3), strides=1, padding='same', if_baisc=False)
    #layer 3
    x1 = basic_bottle(x1, filters=128, kernel_size=(3, 3), strides=2, padding='same', if_baisc=True)
    x1 = basic_bottle(x1, filters=128, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    # layer 4
    x1 = basic_bottle(x1, filters=256, kernel_size=(3, 3), strides=2, padding='same', if_baisc=True)
    x1 = basic_bottle(x1, filters=256, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    # layer 5
    x1 = basic_bottle(x1, filters=512, kernel_size=(3, 3), strides=2, padding='same', if_baisc=True)
    x1 = basic_bottle(x1, filters=512, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    #GlobalAveragePool
    x1 = layers.GlobalAveragePooling2D()(x1)
    e1 = layers.Dense(class_nums, activation='relu')(x1)
    ## applied two BBAs and generate corresponding evidence belief b and uncertainty u.
    b12 = layers.Dense(class_nums, activation='softmax')(x1)

    alpha11 = e1 + 1
    S1 = tf.reduce_sum(alpha11,axis=1,keepdims=True)
    b11 = e1 / S1
    u1 = class_nums/S1

    ## DST fusion for two BBAs
    b1 = b11*b12+u1*b12
    kappa1 = tf.reduce_sum(b1,axis=1,keepdims=True)
    b_new1 = (1-u1)*b1/kappa1
    alpha_new1 = S1*b_new1+1

    ############## For the second modality, construct your DNN
    #layer 1
    x2 = conv2d_bn(inpt2, filters=64, kernel_size=(7,7), strides=2, padding='valid')
    x2 = layers.MaxPool2D(pool_size=(3,3), strides=2)(x2)
    #layer 2
    x2 = basic_bottle(x2, filters=64, kernel_size=(3,3), strides=1, padding='same', if_baisc=False)
    x2 = basic_bottle(x2, filters=64, kernel_size=(3,3), strides=1, padding='same', if_baisc=False)
    #layer 3
    x2 = basic_bottle(x2, filters=128, kernel_size=(3, 3), strides=2, padding='same', if_baisc=True)
    x2 = basic_bottle(x2, filters=128, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    # layer 4
    x2 = basic_bottle(x2, filters=256, kernel_size=(3, 3), strides=2, padding='same', if_baisc=True)
    x2 = basic_bottle(x2, filters=256, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    # layer 5
    x2 = basic_bottle(x2, filters=512, kernel_size=(3, 3), strides=2, padding='same', if_baisc=True)
    x2 = basic_bottle(x2, filters=512, kernel_size=(3, 3), strides=1, padding='same', if_baisc=False)
    #GlobalAveragePool
    x2 = layers.GlobalAveragePooling2D()(x2)
    ## applied two BBAs and generate corresponding evidence belief b and uncertainty u.
    e2 = layers.Dense(class_nums, activation='relu')(x2)
    b22 = layers.Dense(class_nums, activation='softmax')(x2)

    alpha21 = e2 + 1
    S2 = tf.reduce_sum(alpha21,axis=1,keepdims=True)
    b21 = e2 / S2
    u2 = class_nums/S2
    ## DST fusion for two BBAs
    b2 = b21*b22+u2*b22
    kappa2 = tf.reduce_sum(b2,axis=1,keepdims=True)
    b_new2 = (1-u2)*b2/kappa2
    alpha_new2 = S2*b_new2+1

    ############# DST fusion for two modalities
    b = b_new1*b_new2 + b_new1*u2 + b_new2*u1
    u = u1*u2
    kappa = tf.reduce_sum(b,axis=1,keepdims=True)+u
    b_new = b/kappa
    u_new = u/kappa
    S = class_nums/u_new
    alpha_new = S*b_new+1
    model = tf.keras.Model(inputs=(inpt1,inpt2), outputs=alpha_new)
    return model

