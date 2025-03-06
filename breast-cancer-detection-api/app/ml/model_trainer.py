import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from keras.applications import VGG16, VGG19, ResNet50
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score
from sqlalchemy.orm import Session
from app.db import crud
from app.schemas.model import ModelCreate
import os
import shutil


class ModelTrainer:
    def __init__(self, dataset_path, dataset_id, db_session: Session):
        self.dataset_path = dataset_path
        self.db_session = db_session

        # Ensure dataset_id is always an integer string
        self.dataset_id = str(dataset_id)

        # Create consistent directory structure based on dataset_id
        self.models_dir = os.path.join("models", self.dataset_id)
        self.results_dir = os.path.join("results", self.dataset_id)

        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    async def train_all_models(self):
        try:
            benign_path = os.path.join(self.dataset_path, "benign")
            malignant_path = os.path.join(self.dataset_path, "malignant")

            if not (os.path.exists(benign_path) and os.path.exists(malignant_path)):
                print(
                    f"Dataset structure is invalid. Expected 'benign' and 'malignant' folders in {self.dataset_path}")
                return

            # Load and augment data
            X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_augment_data(
                benign_path, malignant_path)

            # Define models to train
            model_builders = {
                'CNN': self.build_cnn,
                'VGG16': self.build_vgg16,
                'VGG19': self.build_vgg19,
                'ResNet': self.build_resnet
            }

            # Train each model
            for model_name, model_builder in model_builders.items():
                try:
                    print(
                        f"Training {model_name} model for dataset ID {self.dataset_id}...")

                    # Build the model
                    model = model_builder(X_train.shape[1:])

                    # Train the model
                    history = self.train_model(
                        model, X_train, y_train, X_val, y_val, model_name)

                    # Evaluate the model
                    test_metrics = self.evaluate_model(
                        model, X_test, y_test, model_name, history)

                    # Save model to file system - directly in dataset_id folder
                    model_path = os.path.join(
                        self.models_dir, f"{model_name}.keras")
                    model.save(model_path)
                    print(f"Model saved to: {model_path}")

                    # Save model to database
                    model_data = ModelCreate(
                        name=model_name,
                        accuracy=float(test_metrics['accuracy']),
                        loss=float(test_metrics['loss']),
                        auc=float(test_metrics['auc']),
                        model_path=model_path,
                        dataset_id=int(self.dataset_id)
                    )

                    try:
                        # Try to find existing model
                        existing_model = self.db_session.query(crud.models.Model).filter(
                            crud.models.Model.name == model_name,
                            crud.models.Model.dataset_id == model_data.dataset_id
                        ).first()

                        if existing_model:
                            # If model exists, delete old model file if path is different
                            if existing_model.model_path and os.path.exists(existing_model.model_path):
                                if existing_model.model_path != model_path:
                                    try:
                                        os.remove(existing_model.model_path)
                                        print(
                                            f"Deleted old model file: {existing_model.model_path}")
                                    except Exception as e:
                                        print(
                                            f"Could not delete old model file: {e}")

                            # Update model in database
                            model_data_dict = model_data.model_dump()
                            for key, value in model_data_dict.items():
                                setattr(existing_model, key, value)
                            print(
                                f"Updated existing {model_name} model in database")
                        else:
                            # Create new model in database
                            model_data_dict = model_data.model_dump()
                            db_model = crud.models.Model(**model_data_dict)
                            self.db_session.add(db_model)
                            print(
                                f"Created new {model_name} model in database")

                        self.db_session.commit()
                        print(
                            f"Model {model_name} trained and saved successfully!")

                    except Exception as db_error:
                        self.db_session.rollback()
                        print(
                            f"Database error while saving model {model_name}: {db_error}")

                except Exception as e:
                    print(f"Error training {model_name}: {e}")

        except Exception as e:
            print(f"Error during model training: {e}")

    def load_and_augment_data(self, benign_path, malignant_path, target_size=(224, 224)):
        """Load and preprocess image data"""
        # Load images
        benign_images = [os.path.join(benign_path, img) for img in os.listdir(benign_path)
                         if os.path.isfile(os.path.join(benign_path, img))]
        malignant_images = [os.path.join(malignant_path, img) for img in os.listdir(malignant_path)
                            if os.path.isfile(os.path.join(malignant_path, img))]

        print(
            f"Found {len(benign_images)} benign and {len(malignant_images)} malignant images")

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

        benign_data, benign_labels = augment_data(
            benign_data, benign_labels, target_count)
        malignant_data, malignant_labels = augment_data(
            malignant_data, malignant_labels, target_count)

        # Merge datasets and split
        X = np.concatenate([benign_data, malignant_data], axis=0)
        y = np.concatenate([benign_labels, malignant_labels], axis=0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train)

        print(
            f"Data split: {len(X_train)} training, {len(X_val)} validation, {len(X_test)} test samples")

        return X_train, X_val, X_test, y_train, y_val, y_test

    # Model architecture definitions remain the same
    def build_cnn(self, input_shape):
        model = Sequential([
            Input(input_shape),
            Conv2D(32, (3, 3), activation='relu'),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(10, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        return model

    def build_vgg16(self, input_shape):
        base_model = VGG16(weights='imagenet',
                           include_top=False, input_shape=input_shape)
        x = Flatten()(base_model.output)
        x = Dropout(0.2)(x)
        x = Dense(512, activation="relu")(x)
        x = Dense(128, activation="relu")(x)
        prediction = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=prediction)
        return model

    def build_vgg19(self, input_shape):
        base_model = VGG19(weights='imagenet',
                           include_top=False, input_shape=input_shape)
        x = Flatten()(base_model.output)
        x = Dropout(0.2)(x)
        x = Dense(512, activation="relu")(x)
        x = Dense(128, activation="relu")(x)
        prediction = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=prediction)
        return model

    def build_resnet(self, input_shape):
        base_model = ResNet50(weights='imagenet',
                              include_top=False, input_shape=input_shape)
        x = Flatten()(base_model.output)
        x = Dropout(0.2)(x)
        x = Dense(512, activation="relu")(x)
        x = Dense(128, activation="relu")(x)
        prediction = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=prediction)
        return model

    def train_model(self, model, X_train, y_train, X_val, y_val, model_name):
        """Train a single model without redundant checkpoints"""
        # Save model directly to its final location
        model_path = os.path.join(self.models_dir, f"{model_name}.keras")

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # Use ModelCheckpoint to save ONLY the best version directly to the final path
        checkpoint = ModelCheckpoint(
            model_path,  # Save directly to final path
            save_best_only=True,
            monitor='val_loss'
        )
        early_stop = EarlyStopping(patience=5, restore_best_weights=True)
        scheduler = ReduceLROnPlateau(factor=0.5, patience=3)

        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=32,
            callbacks=[checkpoint, early_stop, scheduler],
            verbose=1
        )

        return history

    def evaluate_model(self, model, X_test, y_test, model_name, history):
        """Evaluate the model and generate plots"""
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"{model_name} - Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

        y_pred = model.predict(X_test).ravel()
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred)

        # Create model-specific results directory using the standard structure
        model_results_dir = os.path.join(self.results_dir, model_name)
        os.makedirs(model_results_dir, exist_ok=True)

        # Save all plots in the model-specific directory
        # ROC Curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name} (AUC = {auc_score:.3f})')
        plt.legend()
        plt.savefig(os.path.join(model_results_dir, 'roc_curve.png'))
        plt.close()

        # Precision-Recall Curve
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend()
        plt.savefig(os.path.join(model_results_dir, 'precision_recall.png'))
        plt.close()

        # Training vs Validation Accuracy
        plt.figure(figsize=(10, 8))
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(f'Training vs Validation Accuracy - {model_name}')
        plt.legend()
        plt.savefig(os.path.join(model_results_dir, 'training_history.png'))
        plt.close()

        # Confusion matrix
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
        plt.savefig(os.path.join(model_results_dir, 'confusion_matrix.png'))
        plt.close()

        report = classification_report(
            y_test, y_pred_binary, output_dict=True, zero_division=0)

        return {
            'accuracy': test_acc,
            'loss': test_loss,
            'auc': auc_score,
            'precision': report['1']['precision'] if '1' in report else 0,
            'recall': report['1']['recall'] if '1' in report else 0,
            'f1': report['1']['f1-score'] if '1' in report else 0,
        }
