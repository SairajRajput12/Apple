import base64
import io
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import tflite_runtime.interpreter as tflite

# Load the TFLite model
MODEL_PATH = 'apple.tflite'
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

app = Flask(__name__)

def load_and_preprocess_image(image_base64, target_size=(224, 224)):
    try:
        # Decode the base64 image
        image_data = base64.b64decode(image_base64)
        # Convert the binary image to an image object
        img = Image.open(io.BytesIO(image_data))
        print('Image decoded and opened.')

        # Resize the image to (64, 64)
        img = img.resize(target_size)
        print('Image resized to 64x64.')

        # Convert image to numpy array and normalize
        img_array = np.array(img)
        print('Image converted to array.')

        # Normalize the image array
        img_array = img_array.astype('float32') / 255.0
        print('Image normalized.')

        # Add a batch dimension (1, 64, 64, 3)
        img_array = np.expand_dims(img_array, axis=0)
        print('Batch dimension added.')

        return img_array

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise

def predict_image_class(image_base64, class_indices):
    print('Start prediction...')

    # Load and preprocess the image
    preprocessed_img = load_and_preprocess_image(image_base64)
    print('Image preprocessed.')

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], preprocessed_img)

    # Invoke the interpreter to run inference
    interpreter.invoke()

    # Get the prediction output
    predictions = interpreter.get_tensor(output_details[0]['index'])
    print('Prediction made.')

    # Get the predicted class
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_indices[predicted_class_index]

    return predicted_class

@app.route('/')
def home():
    return jsonify({"message": "This is the Plant Disease Detection System!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the base64 image data from the request
        data = request.json
        img_base64 = data.get("img_base64")
        print('Received base64 image data.')

        if not img_base64:
            return jsonify({"error": "No image data provided"}), 400

        print('Processing base64 image data.')

        # Define class labels
        class_labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy']


        # Perform prediction
        prediction = predict_image_class(img_base64, class_labels)
        print(f'Prediction: {prediction}')

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
