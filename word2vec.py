# -*- coding: utf-8 -*-
# __author__ = 'wk'

from __future__ import absolute_import,print_function,division
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 电影数目，及emb维度
movieLens = 4000
embedding_dim = 10
CSV_HEADER=['center','context','output']

def get_dataset_from_csv(csv_file_path, shuffle=False, batch_size=128):
    def process(features,label):
        features = tf.stack(list(features.values()), axis=1)
        return features,label

    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=CSV_HEADER,
        label_name = CSV_HEADER[-1],
        num_epochs=1,
        header=False,
        field_delim="|",
        shuffle=shuffle
    ).map(process)

    return dataset


def word2vec():
    model = keras.Sequential([
        layers.Embedding(movieLens, embedding_dim, input_length=2),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.summary()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model



def word2vec1():
    input = keras.Input(shape=(2,),dtype=tf.int32,name="center")
    x1 = layers.Embedding(output_dim = embedding_dim,
                         input_dim = movieLens,
                         input_length = 2)(input)
    x = layers.Flatten()(x1)
    x = layers.Dense(10,activation="relu")(x)
    output = layers.Dense(1,activation="sigmoid")(x)
    model = keras.Model(inputs = input,outputs = output)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

def word2vec2():
    input1 = keras.Input(shape=(2,),dtype=tf.int32,name="center")
    x = layers.Embedding(output_dim = embedding_dim,
                         input_dim = movieLens,
                         input_length = 2)(input1)
    x = layers.GlobalMaxPool1D()(x)  ##
    x = layers.Dense(10,activation="relu")(x) ##
    output = layers.Dense(1,activation="sigmoid",)(x)
    model = keras.Model(inputs = input1,outputs = output)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

def main():
    model = word2vec1()
    train_dataset = get_dataset_from_csv(r"D:\深度学习\推荐\recommand\train_data.csv", shuffle=True, batch_size=128)
    test_dataset = get_dataset_from_csv(r"D:\深度学习\推荐\recommand\test_data.csv", batch_size=128)

    model.fit(train_dataset, epochs=5,verbose=2,validation_data=test_dataset)

    # Evaluate the model on the test data.
    _, acc = model.evaluate(test_dataset, verbose=0)
    print(acc)

if __name__ == "__main__":
    main()







