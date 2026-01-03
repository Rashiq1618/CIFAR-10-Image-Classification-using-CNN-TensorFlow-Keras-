import cv2
import cv2 as opencv
import numpy as np
import matplotlib.pyplot as plt

from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, datasets, models

#Load dataset
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

# Normalize pixel values
training_images = training_images / 255.0
testing_images = testing_images / 255.0

# Class names
class_names = [
    'Plane', 'Car', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

model = models.load_model('cifar10_super_model.h5')

img = cv2.imread('deer.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f"Prediction is: {class_names[index]}")
plt.show()
