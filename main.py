"""
Brain Tumor Detection System
Main application entry point
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

class BrainTumorDetection:
    def __init__(self, model_path='tumor_detection_model.h5'):
        try:
            self.model = load_model(model_path)
            print("Model loaded successfully!")
        except FileNotFoundError:
            print(f"Model not found at {model_path}")
            print("Please train the model first using: python train_model.py")
    
    def preprocess_image(self, image_path, img_size=(224, 224)):
        """Load and preprocess image"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        img = cv2.resize(img, img_size)
        img = img / 255.0
        return np.expand_dims(img, axis=0)
    
    def predict(self, image_path):
        """Make prediction on single image"""
        img = self.preprocess_image(image_path)
        prediction = self.model.predict(img, verbose=0)
        
        no_tumor_prob = prediction[0][0]
        tumor_prob = prediction[0][1]
        
        result = {
            'tumor_detected': tumor_prob > no_tumor_prob,
            'tumor_probability': float(tumor_prob * 100),
            'confidence': float(max(no_tumor_prob, tumor_prob) * 100)
        }
        
        return result
    
    def batch_predict(self, folder_path):
        """Make predictions on multiple images"""
        results = []
        for img_name in os.listdir(folder_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(folder_path, img_name)
                try:
                    result = self.predict(img_path)
                    result['image'] = img_name
                    results.append(result)
                    print(f"{img_name}: {'TUMOR' if result['tumor_detected'] else 'NO TUMOR'} ({result['confidence']:.2f}%)")
                except Exception as e:
                    print(f"Error processing {img_name}: {str(e)}")
        
        return results

if __name__ == "__main__":
    detector = BrainTumorDetection()
    
    # Example usage
    print("Brain Tumor Detection System")
    print("=" * 50)
    print("To use this system:")
    print("1. Train model: python train_model.py")
    print("2. Predict: python predict.py <image_path>")
    print("=" * 50)