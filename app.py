import base64
import io
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import tflite_runtime.interpreter as tflite
from flask_cors import CORS

# Class labels for Apple and Corn
apple = ['Apple Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy'] 
corn = ['Corn_(maize) Cercospora leaf spot Gray leaf spot', 'Corn (maize) Common rust ', 'Corn (maize) Northern Leaf Blight', 'Corn (maize) healthy']
cherry = ['Cherry (including sour) Powdery mildew','Cherry (including sour) healthy']
grape = ['Grape Black rot', 'Grape Esca (Black_Measles)', 'Grape leaf blight (Isariopsis Leaf Spot)', 'Grape healthy']
peach = ['Peach Bacterial spot', 'Peach healthy']
pepper = ['Pepper bell Bacterial spot', 'Pepper bell healthy']
potato = ['Potato Early blight', 'Potato Late blight', 'Potato healthy']
soyabean = ['Soybean healthy', 'Squash Powdery mildew']
strawberry = ['Strawberry Leaf scorch', 'Strawberry healthy']
tomato = ['Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Target Spot', 'Tomato healthy']





app = Flask(__name__)
CORS(app)  


def load_and_preprocess_image(image_base64, target_size=(224, 224)):
    try:
        # Decode the base64 image
        image_data = base64.b64decode(image_base64)
        # Convert the binary image to an image object
        img = Image.open(io.BytesIO(image_data))
        print('Image decoded and opened.')

        # Resize the image to the model input size (224x224)
        img = img.resize(target_size)
        print('Image resized to 224x224.')

        # Convert image to numpy array and normalize
        img_array = np.array(img)
        print('Image converted to array.')

        # Normalize the image array
        img_array = img_array.astype('float32') / 255.0
        print('Image normalized.')

        # Add a batch dimension (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        print('Batch dimension added.')

        return img_array

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise

def predict_image_class(image_base64, class_indices, model_path):
    print('Start prediction...')

    # Load and preprocess the image
    preprocessed_img = load_and_preprocess_image(image_base64)
    print('Image preprocessed.')

    # Load the TFLite model
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor with preprocessed image
    interpreter.set_tensor(input_details[0]['index'], preprocessed_img)

    # Invoke the interpreter to run inference
    interpreter.invoke()

    # Get the prediction output
    predictions = interpreter.get_tensor(output_details[0]['index'])
    print('Prediction made.')

    # Get the predicted class
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    print(predictions)
    predicted_class = class_indices[predicted_class_index]

    return predicted_class

@app.route('/')
def home():
    return jsonify({"message": "This is the Plant Disease Detection System!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the base64 image data and crop name from the request
        data = request.json
        img_base64 = data.get("img_base64")
        name_of_crop = data.get("name")
        print('Received base64 image data.')

        if not img_base64 or not name_of_crop:
            return jsonify({"error": "No image data or crop name provided"}), 400
    
        print('Processing base64 image data.')

        # Define class labels and model path based on the crop
        if name_of_crop == "corn": 
            class_labels = corn
            MODEL_PATH = 'corn.tflite'
        elif name_of_crop == "apple": 
            class_labels = apple
            MODEL_PATH = 'apple.tflite' 
        elif name_of_crop == "cherry": 
            class_labels = cherry 
            MODEL_PATH = 'cherry.tflite' 

        elif name_of_crop == "grape": 
            class_labels = grape 
            MODEL_PATH = 'grape.tflite'  

        elif name_of_crop == 'peach': 
            class_labels = peach 
            MODEL_PATH = 'peach.tflite' 
        elif name_of_crop == 'pepper': 
            class_labels = pepper  
            MODEL_PATH = 'pepper.tflite'
        elif name_of_crop == 'potato': 
            class_labels = potato 
            MODEL_PATH = 'potato.tflite' 
        elif name_of_crop == 'soyabean': 
            class_labels = soyabean 
            MODEL_PATH = 'soyabean.tflite'  

        elif name_of_crop == 'strawberry': 
            class_labels = strawberry  
            MODEL_PATH = 'strawberry.tflite' 
        elif name_of_crop == 'tomato': 
            class_labels = tomato 
            MODEL_PATH = 'tomato.tflite' 
        else:
            return jsonify({"error": "Invalid crop name provided."}), 400

        # Perform prediction
        prediction = predict_image_class(img_base64, class_labels, MODEL_PATH)
        print(f'Prediction: {prediction}')

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
