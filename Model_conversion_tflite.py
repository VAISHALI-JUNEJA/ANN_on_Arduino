import tensorflow as tf
from tensorflow import keras
import numpy as np
## load model and cpnvert it to tflite
model = keras.models.load_model('ANN_64')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

##Install xxd
# Save the file as a C source file
##!xxd -i tflite_model > g_model.cc
# Print the source file
##!cat g_model.cc
"""put the binary file g_model.cc in model.cpp"""

### Convert the model to the TensorFlow Lite format with quantization
##converter = tf.lite.TFLiteConverter.from_keras_model(model_2)
# Indicate that we want to perform the default optimizations,
# which include quantization
##converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Define a generator function that provides our test data's x values
# as a representative dataset, and tell the converter to use it
##def representative_dataset_generator():
##for value in x_test:
# Each scalar value must be inside of a 2D array that is wrapped in a list
##yield [np.array(value, dtype=np.float32, ndmin=2)]
##converter.representative_dataset = representative_dataset_generator


