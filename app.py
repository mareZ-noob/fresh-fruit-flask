import os
import base64
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
from flask import Flask, request, jsonify
from datetime import datetime

# Suppress warnings and TensorFlow logging
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    """Custom DepthwiseConv2D layer that ignores the 'groups' parameter."""
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        config.pop('groups', None)
        return super().from_config(config)

class CustomSeparableConv2D(tf.keras.layers.SeparableConv2D):
    """Custom SeparableConv2D layer that handles legacy parameters."""
    def __init__(self, *args, **kwargs):
        # Remove incompatible parameters
        kwargs.pop('groups', None)
        kwargs.pop('kernel_initializer', None)
        kwargs.pop('kernel_regularizer', None)
        kwargs.pop('kernel_constraint', None)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        # Remove incompatible parameters from config
        config.pop('groups', None)
        config.pop('kernel_initializer', None)
        config.pop('kernel_regularizer', None)
        config.pop('kernel_constraint', None)
        return super().from_config(config)

def preprocess_base64_image(base64_string):
    """Preprocess base64 encoded image for model prediction and save."""
    try:
        # Remove base64 prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 string to numpy array
        image_bytes = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode original image
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if original_img is None:
            raise ValueError("Failed to decode image from base64 string")
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        original_filename = f"original_image_{timestamp}.jpg"
        processed_filename = f"processed_image_{timestamp}.jpg"
        
        # Save original image
        cv2.imwrite(original_filename, original_img)
        
        # Resize and preprocess for model
        img = cv2.resize(original_img, (100, 100))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Save processed image
        cv2.imwrite(processed_filename, img)
        
        # Normalize for model prediction
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        return img, original_filename, processed_filename
    except Exception as e:
        raise Exception(f"Base64 image preprocessing error: {str(e)}")

def print_freshness(prediction_value):
    """Classify freshness level based on prediction value."""
    threshold_fresh = 0.10
    threshold_medium = 0.35
    
    if prediction_value < threshold_fresh:
        return "FRESH"
    elif prediction_value < threshold_medium:
        return "MEDIUM FRESH"
    else:
        return "NOT FRESH"

def load_legacy_model(model_path='rottenvsfresh98pval.h5'):
    """Load the model with custom objects to handle legacy compatibility."""
    custom_objects = {
        'DepthwiseConv2D': CustomDepthwiseConv2D,
        'SeparableConv2D': CustomSeparableConv2D
    }
    
    try:
        # Clear any existing models/sessions
        tf.keras.backend.clear_session()
        
        # Load the model with custom objects
        model = load_model(model_path, 
                         custom_objects=custom_objects, 
                         compile=False)
        return model
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

# Load the model once when the application starts
global_model = load_legacy_model()

# Create Flask application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_freshness():
    """Endpoint to predict food freshness from base64 encoded image."""
    try:
        # Get JSON data
        request_data = request.get_json()
        
        # Check if image is in the request
        if not request_data or 'image' not in request_data:
            return jsonify({
                'error': 'No image provided',
                'status': 'fail'
            }), 400
        
        # Get base64 image string
        base64_image = request_data['image']
        
        # Preprocess image and save
        processed_img, original_filename, processed_filename = preprocess_base64_image(base64_image)
        
        # Make prediction
        with tf.device('/CPU:0'):  # Force CPU usage to avoid potential GPU issues
            prediction = global_model.predict(processed_img, verbose=0)
        
        # Extract prediction value
        prediction_value = float(prediction[0][0])
        
        # Get freshness classification
        freshness_status = print_freshness(prediction_value)
        
        # Return response
        return jsonify({
            'prediction_value': prediction_value,
            'freshness_status': freshness_status,
            'original_image': original_filename,
            'processed_image': processed_filename,
            'status': 'success'
        })
    
    except Exception as e:
        # Handle any errors during prediction
        return jsonify({
            'error': str(e),
            'status': 'fail'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'message': 'Freshness Classifier API is running'
    })

def setup_gpu():
    """Set up GPU memory growth if available."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == '__main__':
    # Setup GPU if available
    setup_gpu()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)