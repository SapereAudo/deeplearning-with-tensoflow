#! /usr/bin/env python3.6
# -*-coding:utf-8 -*-
# __author__ = "wk"


from __future__ import absolute_import,print_function,division
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,datasets
from tensorflow.keras.layers import Layer,Activation
from sklearn.datasets import load_breast_cancer

class FM(Layer):
    def __init__(self,k,input_dim):
        super(FM,self).__init__()
        self.k = k
        self.weight = self.add_weight(name="weight",
                                      shape=(input_dim,1),
                                      initializer="glorot_uniform",
                                      trainable=True)
        self.bias = self.add_weight(name="bias",
                                    shape=(1,),
                                    initializer="zeros",
                                    trainable=True)
        self.V = self.add_weight(
            name="fm_v",
            shape = (input_dim,self.k),
            initializer = "glorot_uniform",
            trainable = True)
        self.activate = Activation("sigmoid")

    def call(self,inputs):
        linear = tf.add(self.bias,tf.reduce_sum(tf.matmul(inputs,self.weight),1,keepdims=True))
        interactions = 0.5*tf.reduce_sum(
            tf.subtract(
                tf.pow(tf.matmul(inputs,self.V),2),
                tf.matmul(tf.pow(inputs,2),tf.pow(self.V,2))
            ),axis=1,keepdims=True
        )
        return self.activate(tf.add(linear,interactions))

def train():
    data = load_breast_cancer()['data']
    target = load_breast_cancer()['target']

    input = keras.Input(shape=(30,))
    out = FM(10,30)(input)
    model = keras.Model(inputs=input,outputs=out)

    model.compile(optimizer= keras.optimizers.Adam(0.001),
                  loss='binary_crossentropy',
                  metrics=['AUC']
                  )
    model.summary()
    model.fit(data,target,batch_size=1,epochs=10,validation_split=0.2)

if __name__ == '__main__':
    train()

