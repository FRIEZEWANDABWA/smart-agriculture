from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50


def build_resnet50(num_classes: int, image_size: int = 224, trainable_base_layers: int = 0) -> keras.Model:
    """
    trainable_base_layers: unfreeze last N layers of the ResNet50 base (0 = fully frozen base).
    """
    base = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )
    base.trainable = False
    if trainable_base_layers > 0:
        base.trainable = True
        for layer in base.layers[:-trainable_base_layers]:
            layer.trainable = False

    inputs = base.input
    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="resnet50_tl")
