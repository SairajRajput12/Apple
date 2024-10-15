# Plant Disease Detection API

This is a Flask-based API for detecting diseases in various crops using TensorFlow Lite (TFLite) models. The system takes a base64-encoded image of a plant leaf and the name of the crop as input, and returns the predicted class (disease) of the plant based on the image.

## Features
- Supports detection for multiple crops, including apple, corn, cherry, grape, peach, pepper, potato, soyabean, strawberry, and tomato.
- Takes input as base64-encoded image and crop name.
- Utilizes pre-trained TFLite models to predict the disease based on the image.
- Returns the predicted disease class for the given crop.

## Requirements
- Python 3.10.11
- Flask
- Flask-CORS
- Pillow
- NumPy
- tflite-runtime

## Installation

1. Clone this repository.
   ```bash
   git clone https://github.com/your-repo/plant-disease-detection-api.git
   cd plant-disease-detection-api
   ```

2. Install the required packages.
   ```bash
   pip install flask pillow numpy flask-cors tflite-runtime
   ```

3. Ensure you have the TFLite models for each crop in the appropriate format. Place the models (`apple.tflite`, `corn.tflite`, etc.) in the root directory or modify the paths accordingly.

## Usage

1. Run the Flask application:
   ```bash
   python app.py
   ```

2. Send a POST request to `/predict` with the base64-encoded image and the crop name as JSON. For example:

   ```json
   {
     "img_base64": "your_base64_encoded_image_here",
     "name": "apple"
   }
   ```

3. Example response:
   ```json
   {
     "prediction": "Apple___Apple_scab"
   }
   ```

## API Endpoints

### `GET /`
- Returns a welcome message indicating that the API is running.

### `POST /predict`
- **Description:** Predicts the disease class of the provided image for the specified crop.
- **Request Body:**
  - `img_base64`: Base64 encoded image string.
  - `name`: Name of the crop. Supported values: `apple`, `corn`, `cherry`, `grape`, `peach`, `pepper`, `potato`, `soyabean`, `strawberry`, `tomato`.
- **Response:**
  - `prediction`: Predicted class of the plant disease.

## Supported Crops and Class Labels

The API currently supports the following crops and disease classes:

### Apple
- Apple Apple scab
- Apple Black rot
- Apple Cedar apple rust
- Apple healthy

### Corn
- Corn Cercospora leaf spot Gray leaf spot
- Corn Common rust
- Corn Northern Leaf Blight
- Corn healthy

### Cherry
- Cherry Powdery mildew
- Cherry healthy

### Grape
- Grape Black rot
- Grape Esca (Black Measles)
- Grape leaf blight (Isariopsis Leaf Spot)
- Grape healthy

### Peach
- Peach Bacterial spot
- Peach healthy

### Pepper
- Pepper bell Bacterial spot
- Pepper bell healthy

### Potato
- Potato Early blight
- Potato Late blight
- Potato healthy

### Soyabean
- Soybean healthy
- Squash Powdery mildew

### Strawberry
- Strawberry Leaf scorch
- Strawberry healthy

### Tomato
- Tomato Bacterial spot
- Tomato Early blight
- Tomato Target Spot
- Tomato healthy

## Error Handling
- If no image or crop name is provided, the API returns a 400 status code with an error message.
- If an invalid crop name is provided, the API returns a 400 status code with an error message.
- If any internal error occurs, the API returns a 500 status code with the error details.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
This project makes use of TensorFlow Lite for efficient image classification on edge devices.

