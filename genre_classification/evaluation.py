# Load the data from dataset and load the model we have built.
# Make evaluation of the 1D CNN model
import os
from sklearn.preprocessing import LabelBinarizer
import utils
from sklearn.model_selection import train_test_split
from keras.models import load_model

# load data
os.environ['AUDIO_DIR']='../data/fma_small/'
AUDIO_DIR = os.environ.get('AUDIO_DIR')
tracks = utils.load('../data/fma_metadata/tracks.csv')
features = utils.load('../data/fma_metadata/features.csv')

subset = tracks.index[tracks['set', 'subset'] <= 'medium']
tracks = tracks.loc[subset]

# load model
onedCNN_model = load_model('oneDCNN.h5')
onedCNN_model.summary()

# get feature and label
features_all = features.loc[subset]
labels_onehot = LabelBinarizer().fit_transform(tracks['track', 'genre_top'])
features_all=features_all.to_numpy()

# data partition
x_train, x_test, y_train, y_test = train_test_split(features_all, labels_onehot, test_size=0.1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

x_train=x_train.reshape((x_train.shape[0],x_train.shape[1],1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1],1))
x_val=x_val.reshape((x_val.shape[0], x_val.shape[1],1))


def evaluate(model,title=''):
    print("\n")
    print("---------------------------")
    print(title)
    print("---------------------------")
    model.summary()
    train_loss, train_acc = model.evaluate(x_train, y_train)
    val_loss, val_acc = model.evaluate(x_val, y_val)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('\nEvaluation Metric:')
    print('train loss:',train_loss,'  train accuracy:',train_acc)
    print('validation loss:',val_loss, 'validation accuracy:',val_acc)
    print('test loss:',test_loss, '  test accuracy:',test_acc)

# model evaluation
evaluate(onedCNN_model,title='1DCNN')


