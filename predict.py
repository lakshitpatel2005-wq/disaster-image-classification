import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("disaster_model.h5")

classes = ["cyclone","earthquake","fire","flood","normal"]

img = cv2.imread("test.jpg")
img = cv2.resize(img,(224,224))
img = img/255.0
img = np.expand_dims(img,axis=0)

prediction = model.predict(img)

print("Prediction:", classes[np.argmax(prediction)])