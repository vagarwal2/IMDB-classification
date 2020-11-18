from keras.datasets import imdb
from keras import layers
from keras import models
from keras import optimizers
import numpy as np
import data_preparation
import model
(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)
x_train=data_preparation.vectorize_sequences(train_data)
y_train=np.asarray(train_labels).astype('float32')
x_test=data_preparation.vectorize_sequences(test_data)
y_test=np.asarray(test_labels).astype('float32')
x_val=x_train[:10000]
partial_x_train=x_train[10000:]
y_val=y_train[:10000]
partial_y_train=y_train[10000:]
model1=model.model_1()
# For model 1
# history=model1.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))
# For model 2
history=model1.fit(partial_x_train,partial_y_train,epochs=4,batch_size=512,validation_data=(x_val,y_val))
model1.save('../models/model2.h5')
np.save('../models/history2.npy',history.history)