import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import numpy as np

def create_model(input_layer_size):
    # input_layer_size = len(words)
    model = Sequential()
    print "INPUT LAYER SIZE: %s"%input_layer_size

    model.add(Dense(256, activation='relu', input_dim=input_layer_size))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))

    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error',
                optimizer=sgd,
                metrics=['mae'])
    print model.summary()
    return model


def train_model(x_train,y_train,x_test=None,y_test=None):
    default_split = int(0.7*len(x_train[0]))
    
    if(not(x_test and y_test)):
        x_train,x_test = x_train[:default_split], x_train[default_split:]
        y_train,y_test = y_train[:default_split], y_train[default_split:]
    
    model = create_model(len(x_train[0]))
    
    filepath="data/weights/emotion-detection-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # check if it splits the train data as well
    model.fit(x_train, y_train,
            epochs=40,
            batch_size=32,
            # validation_split=0.85,
            callbacks=callbacks_list)
    score = model.evaluate(x_test, y_test, batch_size=32)
    print score

def create_categorical_model(input_layer_size):
    # input_layer_size = len(words)
    model = Sequential()
    print "INPUT LAYER SIZE: %s"%input_layer_size

    model.add(Dense(256, activation='relu', input_dim=input_layer_size))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                optimizer=sgd,
                metrics=['accuracy'])
    print model.summary()
    return model

def train_categorical_model(x_train,y_train,x_test=None,y_test=None):
    # model.fit(x_train, y_train,
    #         epochs=20,
    #         batch_size=128)
    # score = model.evaluate(x_test, y_test, batch_size=128)


    default_split = int(0.7*len(x_train[0]))
    
    if(not(x_test and y_test)):
        x_train,x_test = x_train[:default_split], x_train[default_split:]
        y_train,y_test = y_train[:default_split], y_train[default_split:]
    
    model = create_categorical_model(len(x_train[0]))
    
    filepath="data/weights/emotion-detection-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # check if it splits the train data as well
    model.fit(x_train, y_train,
            epochs=40,
            batch_size=32,
            # validation_split=0.85,
            callbacks=callbacks_list)
    score = model.evaluate(x_test, y_test, batch_size=32)
    print score
    return model