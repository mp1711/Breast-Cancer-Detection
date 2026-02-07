from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi import HTTPException
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from app.core.settings import settings
from app.db import crud
import traceback

router = APIRouter()

PREDICTION_FOLDER = "./predictions"

class Predictor:
    def __init__(self, model_path=None, db_session=None, dataset_id=None):
        self.model_path = model_path
        self.db_session = db_session
        self.dataset_id = dataset_id  # Store the dataset_id
        self.model = None
    
    def load_model(self):
        if self.model is None:
            try:
                # If specific model path is provided, use it directly
                if self.model_path and os.path.exists(self.model_path):
                    print(f"Loading model from specified path: {self.model_path}")
                    self.model = load_model(self.model_path, compile=False)
                    return self.model
                
                # Try to get the best model for this dataset
                if self.dataset_id and self.db_session:
                    dataset = crud.get_dataset(self.db_session, dataset_id=self.dataset_id)
                    if dataset and dataset.best_model_id:
                        best_model = crud.get_model(self.db_session, model_id=dataset.best_model_id)
                        if best_model and best_model.model_path and os.path.exists(best_model.model_path):
                            print(f"Loading dataset best model: {best_model.model_path}")
                            self.model = load_model(best_model.model_path, compile=False)
                            return self.model
                        
                    # Check for best_model_path.txt as a fallback
                    model_path_file = os.path.join(dataset.dataset_path, "best_model_path.txt")
                    if os.path.exists(model_path_file):
                        with open(model_path_file, 'r') as f:
                            model_path = f.read().strip()
                            if os.path.exists(model_path):
                                print(f"Loading model from path in best_model_path.txt: {model_path}")
                                self.model = load_model(model_path, compile=False)
                                return self.model
                
                # Otherwise try to load the best model from database
                if self.db_session:
                    models = crud.get_models(self.db_session)
                    if models:
                        best_model = max(models, key=lambda m: m.accuracy or 0)
                        if best_model and best_model.model_path and os.path.exists(best_model.model_path):
                            print(f"Loading best model from database: {best_model.name} (path: {best_model.model_path})")
                            self.model = load_model(best_model.model_path, compile=False)
                            return self.model
                        else:
                            print(f"Best model path not valid: {getattr(best_model, 'model_path', None)}")
                            self._create_dummy_model()
                    else:
                        print("No models found in database")
                        self._create_dummy_model()
                # Try default path as fallback
                elif os.path.exists(settings.MODEL_PATH):
                    print(f"Loading model from settings path: {settings.MODEL_PATH}")
                    self.model = load_model(settings.MODEL_PATH, compile=False)
                    return self.model
                else:
                    print(f"Model path not found: {self.model_path or settings.MODEL_PATH}")
                    self._create_dummy_model()
            except Exception as e:
                print(f"Error loading model: {e}")
                traceback.print_exc()
                self._create_dummy_model()
        return self.model
    
    def preprocess_image(self, img_path, target_size=(224, 224)):
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def predict(self, img_path):
        try:
            model = self.load_model()
            img_array = self.preprocess_image(img_path)
            
            prediction = model.predict(img_array)[0, 0]
            
            result = "Malignant" if prediction >= 0.5 else "Benign"
            confidence = float(prediction) if prediction >= 0.5 else float(1 - prediction)
            explanation_path = None
            
            if result=="Malignant" :
                # Generate explanation for malignant predictions using Grad-CAM
                explanation_path = self.generate_gradcam_explanation(img_path)
                
            return {
                "result": result,
                "confidence": round(confidence, 2),
                "explanation": explanation_path
            }
        except Exception as e:
            traceback.print_exc()
            return {
                "result": "Error",
                "confidence": 0.0,
                "explanation": None,
                "error": str(e)
            }
    
    def generate_gradcam_explanation(self, img_path, target_size=(224, 224)):
        """
        Generate Grad-CAM heatmap explanation for the prediction.
        Shows which regions of the image were most important for the prediction.
        """
        try:
            model = self.load_model()

            # Load and preprocess image
            img = image.load_img(img_path, target_size=target_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Find the last convolutional layer
            last_conv_layer = None
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower():
                    last_conv_layer = layer
                    break

            if last_conv_layer is None:
                print("No convolutional layer found in model, cannot generate Grad-CAM")
                return None

            # Create a model that maps the input image to the activations of the last conv layer
            # as well as the output predictions
            grad_model = tf.keras.models.Model(
                [model.inputs],
                [last_conv_layer.output, model.output]
            )

            # Compute the gradient of the output with respect to the last conv layer
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                # Get the prediction for malignant class
                loss = predictions[:, 0]

            # Extract the gradients
            grads = tape.gradient(loss, conv_outputs)

            # Pool the gradients over all the axes leaving out the channel dimension
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # Weight the channels by the gradients
            conv_outputs = conv_outputs[0]
            pooled_grads = pooled_grads.numpy()
            conv_outputs = conv_outputs.numpy()

            for i in range(pooled_grads.shape[-1]):
                conv_outputs[:, :, i] *= pooled_grads[i]

            # Average over all the channels to get the heatmap
            heatmap = np.mean(conv_outputs, axis=-1)

            # Normalize the heatmap
            heatmap = np.maximum(heatmap, 0)  # ReLU
            if np.max(heatmap) != 0:
                heatmap = heatmap / np.max(heatmap)

            # Resize heatmap to match original image size
            heatmap = cv2.resize(heatmap, target_size)

            # Load original image for overlay
            img_original = cv2.imread(img_path)
            img_original = cv2.resize(img_original, target_size)

            # Convert heatmap to RGB
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Superimpose the heatmap on original image
            superimposed_img = cv2.addWeighted(img_original, 0.6, heatmap, 0.4, 0)

            # Save the result
            os.makedirs(PREDICTION_FOLDER, exist_ok=True)
            file_name = os.path.basename(img_path).split('.')[0]
            output_path = os.path.join(PREDICTION_FOLDER, f"{file_name}_gradcam.png")

            cv2.imwrite(output_path, superimposed_img)

            return output_path

        except Exception as e:
            traceback.print_exc()
            print(f"Error generating Grad-CAM explanation: {e}")
            return None