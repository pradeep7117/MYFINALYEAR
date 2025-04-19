import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
import os

dataset = pd.read_csv("VoiceData/Voice-ParkinsonDatabase.csv")
dataset.fillna(0, inplace = True)
dataset = dataset.values

Y = dataset[:,2:3].ravel()
X = dataset[:,4:dataset.shape[1]]

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)

X = X.reshape(X.shape[0],X.shape[1],1,1)
print(X.shape)
print(Y.shape)

if os.path.exists('model/voice_model.json'):
    with open('model/voice_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
    classifier.load_weights("model/voice_model_weights.h5")
    classifier._make_predict_function()       
else:
    classifier = Sequential()
    classifier.add(Convolution2D(32, 1, 1, input_shape = (X.shape[1], X.shape[2], X.shape[3]), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 256, activation = 'relu'))
    classifier.add(Dense(output_dim = Y.shape[1], activation = 'softmax'))
    print(classifier.summary())
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = classifier.fit(X, Y, batch_size=4, epochs=200, shuffle=True, verbose=2)
    classifier.save_weights('model/voice_model_weights.h5')            
    model_json = classifier.to_json()
    with open("model/voice_model.json", "w") as json_file:
        json_file.write(model_json)
    f = open('model/voice_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
    f = open('model/voice_history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[9] * 100
    print("Training Model Accuracy = "+str(accuracy))


