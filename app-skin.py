#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
#MODEL_PATH="Desktop/Extra/skin-cancer/model1.json"
#model =tf.keras.models.load_model(MODEL_PATH)

model._make_predict_function()

app=Flask(__name__)
Swagger(app)


# In[ ]:


@app.route('/skincancer')
def home():
    return render_template('index-skin.html')


# In[ ]:




@app.route('/skin-cancer',methods=["POST"])
def predict_note_file():
    """Let's Predict Pneumonia
    Hello
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
    
    
if __name__=='__main__':
    app.run(threaded=False)


# In[ ]:




