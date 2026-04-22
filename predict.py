import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys

# Load model
model = load_model('tumor_detection_model.h5')

def predict_tumor(image_path):
    """Predict if brain tumor is present in MRI image"""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)
    
    if prediction[0][0] > prediction[0][1]:
        result = "No Tumor Detected"
        confidence = prediction[0][0] * 100
    else:
        result = "Tumor Detected"
        confidence = prediction[0][1] * 100
    
    return result, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        image_path = sys.argv[1]
        result, confidence = predict_tumor(image_path)
        print(f"Result: {result}")
        print(f"Confidence: {confidence:.2f}%")