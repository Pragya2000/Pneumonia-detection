#!/usr/bin/env python
# coding: utf-8

# In[6]:


import cv2
from flask import Flask,request,render_template
import flasgger
from flasgger import Swagger
import os
import pandas as pd
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import model_from_json
import tensorflow as tf


MODEL_PATH="pneumonia_detection5.h5"
model1 =tf.keras.models.load_model(MODEL_PATH)

model1._make_predict_function()


json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
#MODEL_PATH="Desktop/Extra/skin-cancer/model1.json"
#model =tf.keras.models.load_model(MODEL_PATH)

model._make_predict_function()

app=Flask(__name__)
Swagger(app)




# In[7]:


@app.route('/')
def new():
    return render_template('main.html')

@app.route('/skincancer')
def welcome1():
    return render_template('index-skin.html')

@app.route('/predict-skin-cancer',methods=["POST"])
def predict_file():
    """Let's Predict Skin Cancer
    Skin Cancer
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """

    img = image.load_img(request.files.get("file"),target_size=(224,224,3))
    img=image.img_to_array(img)
    print(img.shape)
    resized_arr=cv2.resize(img,(224,224))
    resized_arr=resized_arr/255
    resized_arr=resized_arr.reshape(-1,224,224,3)
    A=model.predict(resized_arr)
    
    
    if(A.any()==1):
        return "WE THINK IT IS BENIGN"
    else:
        return "WE THINK IT IS MALIGNANT"


@app.route('/pneumoniax')
def welcome2():
    return render_template('index-pneumonia.html')

"""@app.route('/About')
def About():
    return render_template('About.html')
"""


@app.route('/predict-pneumonia',methods=["POST"])
def predict_note_file():
    """Let's Predict Pneumonia
    Pneumonia
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """

    img = image.load_img(request.files.get("file"),target_size=(200, 200),color_mode="grayscale")
    img=image.img_to_array(img)
    #print(img.shape)
    resized_arr=cv2.resize(img,(200,200))
    #resized_arr=resized_arr/255
    resized_arr=resized_arr.reshape(-1,200,200,1)
    A=model1.predict(resized_arr)
    
    
    if(A[0]==1.):
        return "WE THINK IT IS NORMAL"
    else:
        return "WE THINK IT IS PNEUMONIA"
    
    
if __name__=='__main__':
    app.run(threaded=False)


# In[ ]:





# In[ ]:




