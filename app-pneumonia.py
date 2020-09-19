import cv2
from flask import Flask,request,render_template
import flasgger
from flasgger import Swagger
import os
import pandas as pd
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import tensorflow as tf


MODEL_PATH="pneumonia_detection5.h5"
model =tf.keras.models.load_model(MODEL_PATH)

model._make_predict_function()



app=Flask(__name__)
Swagger(app)

@app.route('/pneumonia')
def welcome():
    return render_template('index-pneumonia.html')

@app.route('/About')
def About():
    return render_template('About.html')



@app.route('/predict_file',methods=["POST"])
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

    img = image.load_img(request.files.get("file"),target_size=(200, 200),color_mode="grayscale")
    img=image.img_to_array(img)
    #print(img.shape)
    resized_arr=cv2.resize(img,(200,200))
    #resized_arr=resized_arr/255
    resized_arr=resized_arr.reshape(-1,200,200,1)
    A=model.predict(resized_arr)
    
    
    if(A[0]==1.):
        return "WE THINK IT IS NORMAL"
    else:
        return "WE THINK IT IS PNEUMONIA"
    
    
if __name__=='__main__':
    app.run()




   



