from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
model = load_model("keras_model.h5", compile=False)
print(model)
