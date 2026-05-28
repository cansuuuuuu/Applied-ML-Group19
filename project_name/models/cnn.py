from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models


class CNNModel:
    """
    Convolutional neural network.
    TensorFlow model architecture.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (64, 64, 3),
        num_classes: int = 10,
        learning_rate: float = 1e-3,
    ) -> None:
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.history: Optional[tf.keras.callbacks.History] = None

        self.compile()

    def _build_model(self) -> tf.keras.Model:
        """Simple CNN architecture with three convolutional layers and two dense layers."""
        model = models.Sequential([
            layers.InputLayer(input_shape=self.input_shape),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(self.num_classes),
        ])
        return model

    def compile(self) -> None:
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def summary(self) -> None:
        self.model.summary()

    def fit(
        self,
        train_images,
        train_labels=None,
        validation_data=None,
        epochs: int = 10,
        batch_size: int = 8,
        verbose: int = 1,
    ) -> tf.keras.callbacks.History:
        """Train the model and return the resulting history."""
        self.history = self.model.fit(
            train_images,
            train_labels,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
        return self.history

    def evaluate(self, test_images, test_labels, verbose: int = 2):
        return self.model.evaluate(test_images, test_labels, verbose=verbose)

    def predict(self, images):
        return self.model.predict(images)

    def plot_history(self) -> None:
        """Plot accuracy curves from training run."""
        if self.history is None:
            raise ValueError("Train the model before calling plot_history().")

        history = self.history.history
        plt.plot(history.get("accuracy", []), label="accuracy")
        if "val_accuracy" in history:
            plt.plot(history["val_accuracy"], label="val_accuracy")

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim([0.0, 1.0])
        plt.legend(loc="lower right")
        plt.show()

    def train_and_evaluate(
        self,
        train_images,
        train_labels,
        test_images,
        test_labels,
        epochs: int = 10,
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
