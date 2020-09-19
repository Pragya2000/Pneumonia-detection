# Pneumonia-detection and Skin-Cancer-detection

ML-Pneumonia-Detector

Introduction:
A Deep Learning powered system that helps in the prediction of Pneumonia with an accuracy of 90.529% test accuracy only. The system compares X-Ray images of human lungs to help predict the disease with the application of Swagger API.

Why?
Pneumonia is lung inflammation caused by infection with virus, bacteria, fungi or other pathogens. According to the National Institutes of Health (NIH), chest x-ray is the best test for pneumonia diagnosis. However, reading x-ray images can be tricky and requires domain expertise and experience. It would be nice if we can just ask a computer to read the images and tell us the results.
Although the project does have a throughput of around 91%(round of), it is still advised to consult a doctor before following any treatment.
This project is just a step towards easing the whole process of consulting a doctor everytime.This project provides a normal person the ability to detect if something is off with their chest x-rays and save them some consultation fee!

How does it function?
This deep learning model is trained and tested on a total of 5863 images. The dataset was collected from Kaggle. After all the pre-processing of data, the data is passed through a neural network which consists of 3 hidden layers with relu activation and an input and an output layer. 
Furthermore with the help of threshold value=0.5 the labels are classified as 1 or 0, 1 being Normal and 0 as Pneumonia.Ultimately the project is deployed using Swagger API or Flask for simple interactiveness.


ML-Skin-Cancer-Detector

Introduction:
This json model was taken from https://www.kaggle.com/fanconic/cnn-for-skin-cancer-detection. It has been further been implemented with Swagger API or Flask. It takes an image of the skin as an input and classifies it as Benign or Malignant.
