import onnxruntime as rt
import numpy as np
from PIL import Image
import io
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlzheimerModel:
    def __init__(self, model_path: str):
        """
        Initialize the ONNX model
        
        Args:
            model_path: Path to the .onnx model file
        """
        try:
            self.sess = rt.InferenceSession(model_path)
            self.input_name = self.sess.get_inputs()[0].name
            self.output_name = self.sess.get_outputs()[0].name
            
            # Get expected input shape
            self.input_shape = self.sess.get_inputs()[0].shape
            logger.info(f"Model loaded successfully. Input shape: {self.input_shape}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def preprocess_image(self, image_file: bytes) -> np.ndarray:
        """
        Preprocess image for model inference
        
        Args:
            image_file: Image file as bytes
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Open image from bytes
            image = Image.open(io.BytesIO(image_file))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize based on model input shape
            # Assuming shape is (1, H, W, C) or (1, C, H, W)
            if len(self.input_shape) == 4:
                if self.input_shape[1] == 3:  # (1, 3, H, W) - channels first
                    target_size = (self.input_shape[3], self.input_shape[2])
                else:  # (1, H, W, 3) - channels last
                    target_size = (self.input_shape[2], self.input_shape[1])
            else:
                target_size = (224, 224)  # Default size
            
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(image).astype(np.float32)
            
            # Normalize to [0, 1]
            img_array = img_array / 255.0
            
            # Add batch dimension if needed
            if len(img_array.shape) == 3:
                img_array = np.expand_dims(img_array, axis=0)
            
            # Convert to channels-first if needed (NCHW format)
            if img_array.shape[-1] == 3 and img_array.shape[1] == 224:
                img_array = np.transpose(img_array, (0, 3, 1, 2))
            
            logger.info(f"Image preprocessed. Shape: {img_array.shape}")
            return img_array
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise

    def predict(self, image_file: bytes) -> dict:
        """
        Run inference on the image
        
        Args:
            image_file: Image file as bytes
            
        Returns:
            Dictionary with predictions and confidence
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_file)
            
            # Run inference
            output = self.sess.run([self.output_name], {self.input_name: processed_image})
            
            # Parse output
            predictions = output[0][0]
            
            logger.info(f"Raw predictions shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}")
            logger.info(f"Raw predictions: {predictions}")
            
            # Model outputs 4 classes for Alzheimer's classification
            # Class mapping (adjust based on your training):
            # 0: Normal/Healthy
            # 1: Mild Cognitive Impairment (MCI)
            # 2: Moderate Alzheimer's
            # 3: Severe Alzheimer's
            
            class_names = ["Normal", "Mild Cognitive Impairment", "Moderate Alzheimer's", "Severe Alzheimer's"]
            class_indices = {0: "normal", 1: "mci", 2: "moderate", 3: "severe"}
            
            # Get the class with highest probability
            predicted_class = np.argmax(predictions)
            confidence = float(predictions[predicted_class])
            
            logger.info(f"Predicted class index: {predicted_class}, confidence: {confidence}")
            
            result = {
                "predicted_class": class_names[predicted_class],
                "class_code": class_indices[predicted_class],
                "confidence": confidence,
                "all_predictions": {
                    class_names[i]: float(predictions[i]) 
                    for i in range(len(predictions))
                }
            }
            
            logger.info(f"Final result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
