import matplotlib.pyplot as plt
SIZE = 128
import glob
import cv2
import seaborn as sns
import pandas as pd
from skimage.filters import sobel
from PIL import Image
import io
import numpy as np
from flask import Flask, request, render_template
import pickle
import os
from pathlib import Path
os.chdir('C:/Users/Lenovo/Desktop/Website') 
app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

import sklearn
model = pickle.load(open('models/random_forest_model.pkl','rb'))
def feature_extractor(dataset):
    x_train = dataset
    image_dataset = pd.DataFrame()
    for image in range(x_train.shape[0]):  
        
        
        df = pd.DataFrame() 
        
        input_img = x_train[image, :,:,:]
        img = input_img
        pixel_values = img.reshape(-1)
        df['Pixel_Value'] = pixel_values           
        num = 1  
        kernels = []
        for theta in range(30): 
            theta = theta / 4. * np.pi
            for sigma in (1, 10):  
                lamda = np.pi/4
                gamma = 0.5
                gabor_label = 'Gabor' + str(num)  
                ksize=9
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                kernels.append(kernel)
                fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                df[gabor_label] = filtered_img  
                print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                num += 1  
                
      
        image_dataset = image_dataset.append(df)
    return image_dataset

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/predict',methods=['GET','POST'])
def predict():
    test_images = []
    temperature = float(request.form.get('temperature'))
    humidity = float(request.form.get('humidity'))
    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    image = Image.open(filepath)
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (SIZE, SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    test_images.append(img)
    test_images = np.array(test_images)
    test_images = test_images/255.0
    test_features = feature_extractor(test_images)
    test_features = np.expand_dims(test_features, axis=0)
    test_for_RF = np.reshape(test_features, (test_images.shape[0], -1))
    test_prediction = model.predict(test_for_RF)
    p1 = test_prediction[0]
    if p1==1:
        if (temperature > 32.5 and  humidity<30):
            prediction="Based on analysis of oral images and other parameters, early signs of oral cancer have been detected, seek onchologist as early as possible for checkup and confirmation."
        else:
            prediction="Based on analysis of oral images and other parameters, no signs of oral cancer were detected.Maintain your oral hygiene."
    else:
        prediction="Based on analysis of oral images and other parameters, no signs of oral cancer were detected.Maintain your oral hygiene."
    
    
    
    
    os.remove(filepath)
    
    return render_template('index.html',prediction_text=prediction)

import waitress
if __name__ == "__main__":
    app.run()
   # from waitress import serve
   # serve(app, host="0.0.0.0", port=8080)

    
    