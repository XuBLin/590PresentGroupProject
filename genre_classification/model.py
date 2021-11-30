# Use 1D CNN to classify 16 genres music
# Dataset should be download from the Internet because of the huge size
import os
from keras import layers, optimizers
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import regularizers
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
import utils
from keras.layers import Flatten
from sklearn.model_selection import train_test_split

# user parameters
epochs = 5
verbose = 1
lr = 0.001  # LEARNING RATE
regularization=regularizers.l2(0.001)

# environment
os.environ['AUDIO_DIR']='../data/fma_small/'
AUDIO_DIR = os.environ.get('AUDIO_DIR')
# get data
tracks = utils.load('../data/fma_metadata/tracks.csv')
features = utils.load('../data/fma_metadata/features.csv')

subset = tracks.index[tracks['set', 'subset'] <= 'medium']
tracks = tracks.loc[subset]

# get feature and label
features_all = features.loc[subset]
features_all=features_all.to_numpy()
label=tracks['track', 'genre_top']
print('\nshow the number of data of each label:')
print(label.value_counts())
labels_onehot = LabelBinarizer().fit_transform(label)
# labels_onehot = pd.DataFrame(labels_onehot, index=tracks.index)

# show one-hot label
# genre_name=['Blues','Classical','Country','Easy Listening','Electronic',
#             'Experimental','Folk','Hip-Hop','Instrumental','International',
#             'Jazz','Old-Time / Historic','Pop','Rock','Soul-RnB','Spoken']
#
# temp=tracks['track', 'genre_top']
# gen=temp.tolist()
# print(temp.value_counts())
# for name in genre_name:
#     index=gen.index(name)
#     print(labels_onehot[index, :])


# data partition-train data 80% ; validation 10% ; test data 10%
x_train, x_test, y_train, y_test = train_test_split(features_all, labels_onehot, test_size=0.1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

x_train=x_train.reshape((x_train.shape[0],x_train.shape[1],1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1],1))
x_val=x_val.reshape((x_val.shape[0], x_val.shape[1],1))

print('\n==================================')
print('All label shape:',labels_onehot.shape)
print('All feature shape:',features_all.shape)
print('\nAfter Data Partition:')
print('train data:',x_train.shape,'   train label:',y_train.shape)
print('validation data:',x_val.shape,'   validation label:',y_val.shape)
print('test data:',x_test.shape,'   test label:',y_test.shape)
print('==================================\n')

# model
model = Sequential()
# model.add(Reshape((-1, 1), input_shape=x_train.shape[1:]))
model.add(layers.Conv1D(200, 32, activation='relu', input_shape=(x_train.shape[1],1)))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(8, 32, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(8, 32, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(Flatten())
model.add(layers.Dense(100,activation='relu'))
model.add(layers.Dense(labels_onehot.shape[1], activation='softmax',kernel_regularizer=regularization))

optimizer = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['acc'])
model.summary()

# train model
history = model.fit(x=x_train, y=y_train,validation_data=(x_val,y_val), epochs=epochs,batch_size=518)
# model.fit_generator(train,batch_size=10, nb_epoch=20)

# save model
model.save('oneDCNN.h5')
# predict test data
y_pred = model.predict(x_test)


def report(history, title='', I_PLOT=True):
    print('\n')
    print(title + ": TEST METRIC (loss,accuracy):", model.evaluate(x_test, y_test, batch_size=2000, verbose=verbose))
    # PLOT HISTORY
    epochs = range(1, len(history.history['loss']) + 1)
    plt.figure()
    plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
    plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')

    plt.plot(epochs, history.history['acc'], 'ro', label='Training acc')
    plt.plot(epochs, history.history['val_acc'], 'r', label='Validation acc')

    plt.title(title)
    plt.legend()
    plt.savefig('HISTORY-' + title + '.png')  # save the figure to file
    if (I_PLOT):
        plt.show()
    plt.close()

# plot train and validation loss/accuracy
report(history, title="1DCNN")