import numpy as np
import keras
import matplotlib.pyplot as plt
import random
import tensorflow as tf 
from tensorflow import keras
from keras.models import Sequential
from keras import initializers
from keras import optimizers
from keras import layers 
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Dropout
from keras.utils.vis_utils import plot_model
from keras.layers import *
from load_data import load_dataset
import pandas as pd


model='CNN-2'

train_x, train_y, test_x, test_y, n_classes, genre = load_dataset(verbose=1, mode="Train", datasetSize=0.75)
# datasetSize = 0.75, this returns 3/4th of the dataset.

# Expand the dimensions of the image to have a channel dimension. (nx128x128) ==> (nx128x128x1)
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)

# Normalize the matrices.
train_x = train_x / 255.
test_x = test_x / 255.

if model=='CNN-1':
    N_channels=1
    input_img = keras.Input(shape=(128, 128, N_channels))    

    layer_1 = layers.Conv2D(47, kernel_size=(3,3),activation='relu', padding='same')(input_img)
    layer_1_output = layers.MaxPooling2D((2, 4), padding='same')(layer_1)

    layer_2 = layers.Conv2D(95, (3, 3), activation='relu', padding='same')(layer_1_output)
    layer_2_output = layers.MaxPooling2D((2, 4), padding='same')(layer_2)

    layer_3 = layers.Conv2D(95, (3, 3), activation='relu', padding='same')(layer_2_output)
    layer_3_output = layers.MaxPooling2D((2, 4), padding='same')(layer_3)

    layer_4 = layers.Conv2D(142, (3, 3), activation='relu', padding='same')(layer_3_output)
    layer_4_output = layers.MaxPooling2D((3, 5), padding='same')(layer_4)

    layer_5 = layers.Conv2D(190, (3, 3), activation='relu', padding='same')(layer_4_output)
    layer_5_output = layers.MaxPooling2D((4, 4), padding='same')(layer_5)

    x= layers.Flatten()(layer_5_output)
    final = layers.Dense(8,  activation='sigmoid')(x)

    model = keras.Model(input_img, final)
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy']);
    model.summary() 
elif model=='RCNN':
    dropout_rate = 0.1
    input_img = keras.Input(shape=(128, 128, 1))  # N_channels

    layer_1 = layers.Conv2D(68, kernel_size=(3, 3), padding='same')(input_img)
    spec_x = BatchNormalization(axis=1)(layer_1)
    spec_x = Activation('relu')(spec_x)
    layer_1_output = layers.MaxPooling2D((2, 2), padding='same')(spec_x)
    layer_1_output = Dropout(dropout_rate)(layer_1_output)


    layer_2 = layers.Conv2D(137, (3, 3), activation='relu', padding='same')(layer_1_output)
    spec_x = BatchNormalization(axis=1)(layer_2)
    spec_x = Activation('relu')(spec_x)
    layer_2_output = layers.MaxPooling2D((3, 3), padding='same')(layer_2)
    layer_2_output = Dropout(dropout_rate)(layer_2_output)


    layer_3 = layers.Conv2D(137, (3, 3), activation='relu', padding='same')(layer_2_output)
    spec_x = BatchNormalization(axis=1)(layer_3)
    spec_x = Activation('relu')(spec_x)
    layer_3_output = layers.MaxPooling2D((4, 4), padding='same')(layer_3)
    layer_3_output = Dropout(dropout_rate)(layer_3_output)

    layer_4 = layers.Conv2D(137, (3, 3), activation='relu', padding='same')(layer_3_output)
    spec_x = BatchNormalization(axis=1)(layer_4)
    spec_x = Activation('relu')(spec_x)
    layer_4_output = layers.MaxPooling2D((4, 4), padding='same')(layer_4)
    layer_4_output = Dropout(dropout_rate)(layer_4_output)
    reshape = keras.layers.Reshape((15, 137))(layer_4_output)

    
    # GRU_layer_1 = layers.Bidirectional(layers.GRU(68), input_shape=(None, 137), return_sequences=True) (layer_4_output)
    GRU_layer_1 = keras.layers.GRU(68, return_sequences=True, return_state=False)(reshape)
    reshape2 = keras.layers.Permute((2, 1))(GRU_layer_1)
    GRU_layer_2 = keras.layers.GRU(1, return_sequences=True, return_state=False)(reshape2)
    flat = keras.layers.Flatten()(GRU_layer_2)
    Dense2 = keras.layers.Dense(8, activation='sigmoid')(flat)
    model = keras.Model(input_img, Dense2)
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()   


elif model=='CNN-2':
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=[7,7], kernel_initializer = initializers.he_normal(seed=1), activation="relu", input_shape=(128,128,1)))
    # Dim = (122x122x64)
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=[2,2], strides=2))
    # Dim = (61x61x64)
    model.add(Conv2D(filters=128, kernel_size=[7,7], strides=2, kernel_initializer = initializers.he_normal(seed=1), activation="relu"))
    # Dim = (28x28x128)
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=[2,2], strides=2))
    # Dim = (14x14x128)
    model.add(Conv2D(filters=256, kernel_size=[3,3], kernel_initializer = initializers.he_normal(seed=1), activation="relu"))
    # Dim = (12x12x256)
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=[2,2], strides=2))
    # Dim = (6x6x256)
    model.add(Conv2D(filters=512, kernel_size=[3,3], kernel_initializer = initializers.he_normal(seed=1), activation="relu"))
    # Dim = (4x4x512)
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=[2,2], strides=2))
    # Dim = (2x2x512)
    model.add(BatchNormalization())
    model.add(Flatten())
    # Dim = (2048)
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Dense(1024, activation="relu", kernel_initializer=initializers.he_normal(seed=1)))
    # Dim = (1024)
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu", kernel_initializer=initializers.he_normal(seed=1)))
    # Dim = (256)
    model.add(Dropout(0.25))
    model.add(Dense(64, activation="relu", kernel_initializer=initializers.he_normal(seed=1)))
    # Dim = (64)
    model.add(Dense(32, activation="relu", kernel_initializer=initializers.he_normal(seed=1)))
    # Dim = (32)
    model.add(Dense(n_classes, activation="softmax", kernel_initializer=initializers.he_normal(seed=1)))
    # Dim = (8)
    model.compile(loss="categorical_crossentropy", optimizer=tf.optimizers.Adam(lr=0.0001), metrics=['accuracy'])
    

history = model.fit(train_x, train_y, epochs=10, verbose=1, validation_split=0.1,callbacks=[csv_logger])

plot_model(model, to_file="Saved_Model/Model_Architecture.jpg")
pd.DataFrame(model.fit(train_x, train_y, epochs=10, verbose=1, validation_split=0.1).history).to_csv("Saved_Model/training_history.csv")
score = model.evaluate(test_x, test_y, verbose=1)
model.save("Saved_Model/Model.h5")




#Displaying curves of loss and accuracy during training
# print(history.history.keys())
acc = history.history['acc']
val_acc = history.history['val_acc']
print(acc)
print(val_acc)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
print(epochs)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('loss_plot.png')