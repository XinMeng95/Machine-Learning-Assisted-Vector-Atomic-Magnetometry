# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 16:21:29 2023

@author: arthu
"""

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


predatapath = " " #  path to NNdata
datapath = " " #     path for preprocessed data
datasavepath = " "
def get_file_list(file_path):

    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:

        dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))

        return dir_list


f_list = get_file_list(predatapath)

for filename in f_list:
    if 'theta' in filename:
        a = filename.split()
        theta = int(a[0].split('=')[1])
        phi = int(a[1].split('=')[1].split('.')[0])
        data = np.loadtxt(predatapath + filename,dtype=float, delimiter =',')
        for i in range(1, 69):
            with open(datapath+'label.txt','a') as f:
                f.write('%5f,%5f,%5f\n' %(theta/180,phi/180,(data[i][4]-997)/9))
            with open(datapath+'measure.txt','a') as f:
                f.write('%5f,%5f,%5f,%5f\n' %(100*data[i][0],100*data[i][1],100*data[i][2],100*data[i][3]))


x_train = np.loadtxt(datapath+'measure.txt', dtype=float, delimiter=',')
y_train = np.loadtxt(datapath+'label.txt', dtype=float, delimiter=',')



np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
    tf.keras.layers.Dense(128, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.00001)),
    tf.keras.layers.Dense(3)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.mean_squared_error,
              metrics=['mae'])

checkpoint_save_path = datasavepath+"checkpoint\\vecmag.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)



history = model.fit(x_train, y_train, batch_size=64, epochs=5000, validation_split=0.2, verbose=2, validation_freq=1
                    , callbacks=[cp_callback])

model.summary()
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()