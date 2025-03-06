from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi import HTTPException
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
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
            
            # Check if the model expects the right input shape
            expected_shape = tuple(model.input.shape[1:])
            actual_shape = tuple(img_array.shape[1:])
            
            if expected_shape != actual_shape:
                print(f"Warning: Model expects input shape {expected_shape} but got {actual_shape}")
                # Resize or reshape if needed
            
            prediction = model.predict(img_array)[0, 0]
            
            result = "Malignant" if prediction >= 0.5 else "Benign"
            confidence = float(prediction) if prediction >= 0.5 else float(1 - prediction)
            explanation_path = None
            
            # Generate explanation for all predictions
            explanation_path = self.generate_lime_explanation(img_path)
                
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
    
    def generate_lime_explanation(self, img_path, target_size=(224, 224)):
        try:
            model = self.load_model()
            img = image.load_img(img_path, target_size=target_size)
            img_array = image.img_to_array(img) / 255.0
            
            # Define prediction function that LIME can use (expects batch input)
            def predict_fn(images):
                # Normalize if needed
                return model.predict(images)
            
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                img_array,
                predict_fn,
                top_labels=1,
                hide_color=0,
                num_samples=50
            )
            
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0], 
                positive_only=True, 
                num_features=5, 
                hide_rest=False
            )
            
            os.makedirs(PREDICTION_FOLDER, exist_ok=True)
            file_name = os.path.basename(img_path).split('.')[0]
            output_path = os.path.join(PREDICTION_FOLDER, f"{file_name}_lime.png")
            
            plt.figure(figsize=(8, 8))
            plt.imshow(mark_boundaries(temp, mask))
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            return output_path
        except Exception as e:
            traceback.print_exc()
            print(f"Error generating LIME explanation: {e}")
            return None