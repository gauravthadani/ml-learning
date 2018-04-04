import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

#IMDB Dataset loading
train, test, _ = imdb.load_data(path='imdb.pkl',n_words= 10000, valid_portion=0.1)

trainX, trainY = train
testX, testY = test  

#Data preprocessing
#Sequence padding
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testY = pad_sequences(testY, maxlen=100, value=0.)
#converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)