import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ModelEvaluator:
    """Handles model evaluation and plot generation for both trained and uploaded models"""

    def __init__(self, dataset_id, db_session=None):
        self.dataset_id = dataset_id
        self.db_session = db_session
        self.results_dir = f"results/{dataset_id}"
        os.makedirs(self.results_dir, exist_ok=True)

    def load_test_data(self, dataset_path, target_size=(224, 224)):
        """Load and preprocess test data from dataset directory"""
        benign_path = os.path.join(dataset_path, 'benign')
        malignant_path = os.path.join(dataset_path, 'malignant')

        # Load images
        benign_images = [os.path.join(benign_path, img) for img in os.listdir(benign_path)
                         if os.path.isfile(os.path.join(benign_path, img))]
        malignant_images = [os.path.join(malignant_path, img) for img in os.listdir(malignant_path)
                            if os.path.isfile(os.path.join(malignant_path, img))]

        print(f"Found {len(benign_images)} benign and {len(malignant_images)} malignant images")

        # Load and preprocess images
        def process_images(image_paths, label):
            images, labels = [], []
            for img_path in image_paths:
                try:
                    img = load_img(img_path, target_size=target_size)
                    img = img_to_array(img) / 255.0
                    images.append(img)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    continue
            return np.array(images), np.array(labels)

        benign_data, benign_labels = process_images(benign_images, 0)
        malignant_data, malignant_labels = process_images(malignant_images, 1)

        # Augment data to balance dataset
        max_count = max(len(benign_data), len(malignant_data))
        target_count = int(1.5 * max_count)

        datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2,
                                     height_shift_range=0.2, horizontal_flip=True)

        def augment_data(data, labels, target_count):
            augmented_images, augmented_labels = list(data), list(labels)
            while len(augmented_images) < target_count:
                for img, label in datagen.flow(data, labels, batch_size=1):
                    augmented_images.append(img[0])
                    augmented_labels.append(label[0])
                    if len(augmented_images) >= target_count:
                        break
            return np.array(augmented_images), np.array(augmented_labels)

        benign_data, benign_labels = augment_data(benign_data, benign_labels, target_count)
        malignant_data, malignant_labels = augment_data(malignant_data, malignant_labels, target_count)

        # Merge datasets and split
        X = np.concatenate([benign_data, malignant_data], axis=0)
        y = np.concatenate([benign_labels, malignant_labels], axis=0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y)

        print(f"Test set: {len(X_test)} samples")

        return X_test, y_test

    def evaluate_model(self, model, X_test, y_test, model_name, include_training_history=False, history=None):
        """
        Evaluate the model and generate plots.

        Args:
            model: Trained Keras model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model (CNN, VGG16, etc.)
            include_training_history: Whether to generate training history plot
            history: Training history object (required if include_training_history=True)

        Returns:
            Dict with evaluation metrics
        """
        # Get predictions
        y_pred = model.predict(X_test).ravel()

        # Calculate metrics manually
        y_pred_binary = (y_pred > 0.5).astype(int)
        test_acc = np.mean(y_pred_binary == y_test)

        # Calculate binary crossentropy loss manually
        epsilon = 1e-7  # Small value to avoid log(0)
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        test_loss = -np.mean(y_test * np.log(y_pred_clipped) + (1 - y_test) * np.log(1 - y_pred_clipped))

        print(f"{model_name} - Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

        # Calculate ROC and AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred)

        # Create model-specific results directory
        model_results_dir = os.path.join(self.results_dir, model_name)
        os.makedirs(model_results_dir, exist_ok=True)

        # Generate plots
        self._plot_roc_curve(fpr, tpr, auc_score, model_name, model_results_dir)
        self._plot_precision_recall(precision, recall, model_name, model_results_dir)
        self._plot_confusion_matrix(y_test, y_pred, model_name, model_results_dir)

        # Optionally generate training history plot
        if include_training_history and history is not None:
            self._plot_training_history(history, model_name, model_results_dir)

        # Get classification report
        y_pred_binary = (y_pred > 0.5).astype(int)
        report = classification_report(y_test, y_pred_binary, output_dict=True, zero_division=0)

        return {
            'accuracy': test_acc,
            'loss': test_loss,
            'auc': auc_score,
            'precision': report['1']['precision'] if '1' in report else 0,
            'recall': report['1']['recall'] if '1' in report else 0,
            'f1': report['1']['f1-score'] if '1' in report else 0,
        }

    def _plot_roc_curve(self, fpr, tpr, auc_score, model_name, output_dir):
        """Generate and save ROC curve plot"""
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name} (AUC = {auc_score:.3f})')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()

    def _plot_precision_recall(self, precision, recall, model_name, output_dir):
        """Generate and save Precision-Recall curve plot"""
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'precision_recall.png'))
        plt.close()

    def _plot_training_history(self, history, model_name, output_dir):
        """Generate and save training history plot"""
        plt.figure(figsize=(10, 8))
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Training vs Validation Accuracy - {model_name}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'training_history.png'))
        plt.close()

    def _plot_confusion_matrix(self, y_test, y_pred, model_name, output_dir):
        """Generate and save confusion matrix plot"""
        y_pred_binary = (y_pred > 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred_binary)

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.colorbar()
        tick_marks = [0, 1]
        plt.xticks(tick_marks, ['Benign', 'Malignant'], rotation=45)
        plt.yticks(tick_marks, ['Benign', 'Malignant'])

        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
