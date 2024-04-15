import base64
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import numpy as np
from PIL import Image
import gradio as gr
import tensorflow as tf

# Load the trained model
modelCNN = tf.keras.models.load_model('modelCNN.h5')

def predict(sketch):
    try:
        if 'layers' in sketch:
            # Access the sketch image from the 'layers' key
            sketch_image = np.array(sketch['layers'][0]) * 255
            sketch_image = Image.fromarray(sketch_image.astype('uint8')).convert('L')

            # Resize the sketch to match MNIST image size
            sketch_image = sketch_image.resize((28, 28))

            # Convert the image to a numpy array
            img_array = np.array(sketch_image).reshape(1, 28, 28, 1) / 255.0

            # Make prediction
            prediction = modelCNN.predict(img_array)[0]
            predicted_digit = np.argmax(prediction)
            return str(predicted_digit)
        else:
            # Handle the case where "layers" key is missing
            return "No sketch data found"
    except Exception as e:
        # Print the exception for debugging
        print("Error:", e)
        # Return an error message
        return "An error occurred"

# Launch Gradio interface with FastAPI integration
gr.Interface(fn=predict, inputs="sketchpad", outputs="label").launch()
