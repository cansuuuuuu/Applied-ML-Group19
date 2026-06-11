from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow.keras import layers, models


class NetCloud:
    """
    Model based on the CloudNet architecture from https://doi.org/10.1029/2018GL077787Digital
    This model is based on an Alexnet, although it has one fewer convolutional layers. Originally it was
    meant to be trained on 11 outputs instead of 7 so the performance is undoubtably lower because of that.

    """

    def __init__(self, num_classes: int = 10, image_size : int = 227) -> None:
        self.num_classes = num_classes
        self.image_size = image_size
        self.preprocessing = self._build_preprocessing()
        self.model = self._build_model()
        self.history : Optional[tf.keras.callbacks.History] = None 

        self.compile()

    def _build_preprocessing(self):
        """
        Build the preprocessing pipeline to include it directly into the model
        """
        resize_size = 256
        preprocessing = tf.keras.Sequential([
            layers.Resizing(resize_size, resize_size),
            layers.Cropping2D(((resize_size - self.image_size)//2, (resize_size - self.image_size)//2)),
            # layers.Normalization(),
            layers.Rescaling(1.0 / 255.0)
        ])
        return preprocessing


    def _build_model(self):
        """Alexnet based cloud recognition model architcture"""

        model = models.Sequential([
            layers.Input(shape=(self.image_size, self.image_size, 3)),
            self.preprocessing,
            layers.Conv2D(96, (11,11), strides = 4, input_shape = (self.image_size,self.image_size,3), activation = 'relu'),
            layers.MaxPooling2D((3,3), strides = 2),
            layers.Conv2D(256, (5,5), padding = 'same', activation = 'relu'),
            layers.MaxPooling2D((3,3), strides = 2),
            layers.Conv2D(384,(3,3), padding = 'same', activation = 'relu' ),
            layers.Conv2D(256,(3,3), padding = 'same', activation = 'relu'),
            layers.MaxPooling2D((3,3), strides = 2),
            layers.Flatten(),
            layers.Dense(4096, activation = 'relu'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation = 'relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation = 'softmax')
        ])

        return model

    def compile(self) -> None:
        self.model.compile(
            optimizer = 'adam',
            loss = tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics = ['accuracy']
        )

    def summary(self) -> None:
        self.model.summary()

    def fit(
        self,
        train_images,
        train_labels=None,
        validation_data=None,
        epochs: int = 30,
        batch_size: int = 8,
        verbose: int = 1,
        class_weight=None,
        callbacks=None,
    ) -> tf.keras.callbacks.History:
        """Train the model and return the resulting history."""
        self.history = self.model.fit(
            train_images,
            train_labels,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            class_weight=class_weight,
            callbacks=callbacks,
        )
        return self.history

    def evaluate(self, test_images: np.ndarray, test_labels = None, verbose: int = 2):
        return self.model.evaluate(test_images, test_labels, verbose=verbose)

    def predict(self, images):
        return self.model.predict(images)


    def plot_confusion_matrix(
            self,
            dataset,
            class_names: Optional[Sequence[str]] = None,
            save_path: str = "project_name/data/confusion_matrix.png",
    ) -> None:
        y_true = []
        y_pred = []

        for images, labels in dataset:
            logits = self.model.predict(images, verbose=0)
            y_pred.extend(np.argmax(logits, axis=1))
            y_true.extend(labels.numpy().astype(int).ravel())
    
        labels = list(range(self.num_classes))
        matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
        display_labels = class_names if class_names is not None else labels

        fig, ax = plt.subplots(figsize=(9, 7))
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=display_labels,
        )
        display.plot(
            ax=ax,
            cmap="Blues",
            values_format=".2f",
            xticks_rotation=45,
            colorbar=True,
        )
        ax.set_title("Normalized Confusion Matrix")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")
        plt.show()


    def plot_history(self, save_path : str = "project_name/data/plot_history.png") -> None:
        if self.history is None:
            raise ValueError("Train the model before calling plot_history().")

        history = self.history.history

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history["loss"], label="train_loss")
        if "val_loss" in history:
            axes[0].plot(history["val_loss"], label="val_loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss")
        axes[0].legend()

        axes[1].plot(history["accuracy"], label="train_accuracy")
        if "val_accuracy" in history:
            axes[1].plot(history["val_accuracy"], label="val_accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.show()


    def train_and_evaluate(
            self,
            train_images,
            train_labels,
            test_images,
            test_labels,
            epochs: int = 30,
            batch_size: int = 32,
            verbose: int = 1,
    ):
        """Train, plot, and evaluate the model."""
        self.fit(
            train_images,
            train_labels,
            validation_data=(test_images, test_labels),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.plot_history()
        return self.evaluate(test_images, test_labels, verbose=2)