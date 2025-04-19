from tkinter import messagebox
from tkinter import *
from PIL import Image, ImageTk
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import cv2
import random
import numpy as np
from keras.utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt


main = tkinter.Tk()
main.title("ParkinsonNet: Convolutional Neural Networks Model for Parkinson Disease Detection from Images and Voice Data")
main.geometry("1300x1200")

global filename
global voice_classifier, image_classifier

def loadMLModels():
    global voice_classifier, image_classifier
    with open('model/images_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        image_classifier = model_from_json(loaded_model_json)
    json_file.close()    
    image_classifier.load_weights("model/images_model_weights.h5")
    image_classifier._make_predict_function()

    with open('model/voice_model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        voice_classifier = model_from_json(loaded_model_json)
    json_file.close()    
    voice_classifier.load_weights("model/voice_model_weights.h5")
    voice_classifier._make_predict_function()       


def imageDetection():
    global image_classifier
    labels = ['Healthy','Parkinson']
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = image_classifier.predict(img)
    predict = np.argmax(preds)

    img = cv2.imread(filename)
    img = cv2.resize(img, (500,400))
    cv2.putText(img, 'Image Data Predicted as : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow('Image Data Predicted as : '+labels[predict], img)
    cv2.waitKey(0)


def voiceDetection():
    text.delete('1.0', END)
    global voice_classifier
    labels = ['Healthy','Parkinson']
    filename = filedialog.askopenfilename(initialdir="testVoice")

    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    text.insert(END,"VOICE DATA SAMPLES\n\n")
    text.insert(END,str(dataset.head()))
    dataset = dataset.values
    X = dataset[:,3:dataset.shape[1]]
    X = X.reshape(X.shape[0],X.shape[1],1,1)
    predict = voice_classifier.predict(X)
    predict = np.argmax(predict)
    messagebox.showinfo("Uploaded Voice Data Predicted as "+labels[predict],"Uploaded Voice Data Predicted as "+labels[predict])


def graph():
    f = open('model/images_history.pckl', 'rb')
    image = pickle.load(f)
    f.close()

    f = open('model/voice_history.pckl', 'rb')
    voice = pickle.load(f)
    f.close()

    img_accuracy = image['accuracy']
    img_loss = image['loss']

    voice_accuracy = voice['accuracy']
    voice_loss = voice['loss']
    voice_accuracy = voice_accuracy[150:200]
    voice_loss = voice_loss[150:200]

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy/Loss')
    plt.plot(img_accuracy, 'ro-', color = 'red')
    plt.plot(img_loss, 'ro-', color = 'green')
    plt.plot(voice_accuracy, 'ro-', color = 'blue')
    plt.plot(voice_loss, 'ro-', color = 'orange')
    plt.legend(['Training Accuracy (Image)', 'Training Loss (Image)', 'Training Accuracy (Voice)','Training Loss (Voice)'], loc='upper left')
    plt.title('Performance Comparison of Image and Voice (Training Accuracy, and Loss)')
    plt.show()

def close():
    main.destroy()



b_img=ImageTk.PhotoImage(file="bg.png")
background = Label(main, image=b_img)
#background.pack(fill=BOTH,expand=YES)
background.place(x=0,y=0)
    

font = ('times', 16, 'bold')
title = Label(main, text='ParkinsonNet: Convolutional Neural Networks Model for Parkinson Disease Detection from Images and Voice Data',anchor=W, justify=CENTER)
title.config(bg='yellow4', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 13, 'bold')
imageButton = Button(main, text="Detect Parkinson from Images", command=imageDetection)
imageButton.place(x=50,y=100)
imageButton.config(font=font1)  

voiceButton = Button(main, text="Detect Parkinson from Voice Samples", command=voiceDetection)
voiceButton.place(x=50,y=200)
voiceButton.config(font=font1)

graphButton = Button(main, text="Machine Learning Performance Graph", command=graph)
graphButton.place(x=50,y=250)
graphButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50,y=300)
exitButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=78)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)

loadMLModels()
main.config(bg='Sky Blue')

main.mainloop()
