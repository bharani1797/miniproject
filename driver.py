import cv2
import numpy as np
from keras import backend as K
import keras as keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from keras import applications
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import load_model
import shutil
import os
from PIL import Image
from keras.preprocessing import image as I
from glob import glob
import json
def predict_shape(img):
	img = I.load_img(img, target_size=(299, 299))
	model = load_model("shapes.h5")
	x = I.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	preds = model.predict(x)
	return np.argmax(preds[0])
	
def preprocess_eye(img):
	im = cv2.imread(img)
	img = Image.fromarray(im, 'RGB')
	image = img.resize((100, 100))
	ar= np.array(image)
	ar=ar/255
	label=1
	a=[]
	a.append(ar)
	a=np.array(a)
	return a

def preprocess_chest(img):
	im = cv2.imread(img)
	img = Image.fromarray(im, 'RGB')
	image = img.resize((50, 50))
	ar= np.array(image)
	ar=ar/255
	label=1
	a=[]
	a.append(ar)
	a=np.array(a)
	return a	

def predict_grade(image):
	shape = predict_shape(image)
	if shape == 0:
		model = load_model("cxray.h5")
		prediction = np.argmax(model.predict(preprocess_chest(image)))
		if prediction == 0:
			return "LOW"
		if prediction == 1:
			return "HIGH"
		
	if shape == 1:
		model = load_model("oct.h5")
		prediction = np.argmax(model.predict(preprocess_eye(image)))
		if prediction == 0:
			return "LOW"
		if prediction == 1:
			return "HIGH"
		if prediction == 2:
			return "MEDIUM"
		
global_list = []
high_priority = []
mid_priority = []
low_priority = []
users = sorted(os.listdir("/home/adirao/Desktop/ML/Users"))
print users

for user in users:
	user_dict = {}
	print user
	info_data = glob("/home/adirao/Desktop/ML/Users/"+user+"/*.json")
	print info_data
	with open(info_data[0]) as json_file:
			my_file = json.load(json_file)
			name = my_file["name"]
			age = my_file["age"]
			gender = my_file["gender"]
	user_dict["reference"] = user
	user_dict["name"] = name
	user_dict["age"] = age
	user_dict["gender"] = gender
	image_data = glob("/home/adirao/Desktop/ML/Users/"+user+"/*.jpeg")
	print image_data
	flag = 0
	for image in image_data:
		grade = predict_grade(image)
		if grade == "HIGH":
			user_dict["priority"] = "HIGH"
			flag = 1
		if grade == "MEDIUM" and flag != 1:
			user_dict["priority"] = "MEDIUM"
			flag = 2
		if grade == "LOW" and flag != 1 and flag != 2:
			user_dict["priority"] = "LOW"
	global_list.append(user_dict)		
print global_list
for user in global_list:
	if user["priority"] == "HIGH":
		high_priority.append(user["reference"])
	if user["priority"] == "MEDIUM":
		mid_priority.append(user["reference"])
	else:
		low_priority.append(user["reference"])

for user in high_priority:
		shutil.copytree("/home/adirao/Desktop/ML/Users/"+user,"/home/adirao/Desktop/ML/HIGH/Users")
for user in mid_priority:
		shutil.copytree("/home/adirao/Desktop/ML/Users/"+user,"/home/adirao/Desktop/ML/MED/Users")
for user in low_priority:
		shutil.copytree("/home/adirao/Desktop/ML/Users/"+user,"/home/adirao/Desktop/ML/LOW/Users")

	
	
