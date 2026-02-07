import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class ModelValidator:
    """Validates uploaded model files for compatibility and correctness"""

    EXPECTED_INPUT_SHAPE = (None, 224, 224, 3)
    EXPECTED_OUTPUT_SHAPE = (None, 1)
    VALID_MODEL_TYPES = ['CNN', 'VGG16', 'VGG19', 'ResNet']

    @staticmethod
    def validate_file_extension(filename):
        """Check if file has valid extension (.keras or .h5)"""
        return filename.lower().endswith(('.keras', '.h5'))

    @staticmethod
    def convert_to_keras(file_path):
        """
        Convert .h5 files to .keras format if needed.

        Args:
            file_path: Path to the model file

        Returns:
            Path to .keras file (may be same as input if already .keras)
        """
        if file_path.endswith('.keras'):
            return file_path

        if file_path.endswith('.h5'):
            # Load .h5 model
            model = tf.keras.models.load_model(file_path, compile=False)

            # Save as .keras
            keras_path = file_path.replace('.h5', '.keras')
            model.save(keras_path)

            print(f"Converted {file_path} to {keras_path}")
            return keras_path

        raise ValueError(f"Unsupported file format: {file_path}")

    @staticmethod
    def validate_architecture(model_path):
        """
        Validate that the model has the expected architecture.

        Args:
            model_path: Path to the .keras model file

        Returns:
            Tuple of (success: bool, error_message: str or None)
        """
        try:
            # Load model without compiling
            model = tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            return False, f"Failed to load model file. The file may be corrupted or not a valid Keras model: {str(e)}"

        try:
            # Check input shape
            input_shape = model.input_shape
            if input_shape != ModelValidator.EXPECTED_INPUT_SHAPE:
                return False, f"Invalid input shape: expected {ModelValidator.EXPECTED_INPUT_SHAPE}, got {input_shape}"

            # Check output shape
            output_shape = model.output_shape
            if output_shape != ModelValidator.EXPECTED_OUTPUT_SHAPE:
                return False, f"Invalid output shape: expected {ModelValidator.EXPECTED_OUTPUT_SHAPE}, got {output_shape}"

            print(f"Model architecture validated successfully: input={input_shape}, output={output_shape}")
            return True, None

        except Exception as e:
            return False, f"Error validating model architecture: {str(e)}"

    @staticmethod
    def test_prediction(model_path, dataset_path):
        """
        Test the model by running a prediction on a sample image.

        Args:
            model_path: Path to the .keras model file
            dataset_path: Path to the dataset directory

        Returns:
            Tuple of (success: bool, error_message: str or None)
        """
        try:
            # Load model
            model = tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            return False, f"Failed to load model for testing: {str(e)}"

        try:

            # Find a sample image from benign folder
            benign_path = os.path.join(dataset_path, 'benign')
            if not os.path.exists(benign_path):
                return False, f"Dataset benign folder not found: {benign_path}"

            image_files = [f for f in os.listdir(benign_path) if os.path.isfile(os.path.join(benign_path, f))]
            if not image_files:
                return False, "No images found in benign folder for testing"

            # Load and preprocess a sample image
            sample_image_path = os.path.join(benign_path, image_files[0])
            img = load_img(sample_image_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Try prediction
            prediction = model.predict(img_array, verbose=0)

            # Check prediction shape
            if prediction.shape != (1, 1):
                return False, f"Invalid prediction shape: expected (1, 1), got {prediction.shape}"

            # Check prediction value is in [0, 1] range
            if not (0 <= prediction[0][0] <= 1):
                return False, f"Prediction value out of range [0, 1]: {prediction[0][0]}"

            print(f"Test prediction successful: {prediction[0][0]:.4f}")
            return True, None

        except Exception as e:
            return False, f"Failed during test prediction: {str(e)}"

    @staticmethod
    def validate_model_type(model_type):
        """Validate that model_type is one of the supported types"""
        if model_type not in ModelValidator.VALID_MODEL_TYPES:
            return False, f"Invalid model type: {model_type}. Must be one of {ModelValidator.VALID_MODEL_TYPES}"
        return True, None

    @staticmethod
    def validate_all(file_path, model_type, dataset_path):
        """
        Run all validations on an uploaded model.

        Args:
            file_path: Path to uploaded model file
            model_type: Type of model (CNN, VGG16, VGG19, ResNet)
            dataset_path: Path to dataset directory

        Returns:
            Tuple of (success: bool, error_message: str or None, keras_path: str or None)
        """
        # Validate model type
        success, error = ModelValidator.validate_model_type(model_type)
        if not success:
            return False, error, None

        # Validate file extension
        if not ModelValidator.validate_file_extension(os.path.basename(file_path)):
            return False, "Invalid file format. Only .keras and .h5 files are supported", None

        # Convert to .keras if needed
        try:
            keras_path = ModelValidator.convert_to_keras(file_path)
        except Exception as e:
            return False, f"Failed to convert model: {str(e)}", None

        # Validate architecture
        success, error = ModelValidator.validate_architecture(keras_path)
        if not success:
            return False, error, None

        # Test prediction
        success, error = ModelValidator.test_prediction(keras_path, dataset_path)
        if not success:
            return False, error, None

        return True, None, keras_path
