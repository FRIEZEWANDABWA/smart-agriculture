from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input

from ..preprocess.pipeline import preprocess_bgr_to_rgb


def get_img_array(img_path: Path, skip_preprocess: bool, size: int):
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        raise SystemExit(f"Failed to read image at {img_path}")
    if not skip_preprocess:
        rgb = preprocess_bgr_to_rgb(bgr)
    else:
        resized = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    return np.expand_dims(rgb, axis=0).astype(np.float32)


def make_gradcam_heatmap(img_array: np.ndarray, model: keras.Model, last_conv_layer_name: str) -> tuple[np.ndarray, int]:
    # Form mathematical model to map inputs to final convolution layer and absolute output
    grad_model = keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Compute gradient of the top class with respect to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Pool gradients over height and width axes
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    # Multiply each channel by its absolute importance pooling
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), pred_index.numpy()


def save_and_display_gradcam(img_path: Path, heatmap: np.ndarray, out_path: Path, alpha: float = 0.4) -> None:
    img = cv2.imread(str(img_path))
    # Conform mathematically small heatmap to original raw HxW geometry
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # Convert heatmap topology to RGB
    heatmap = np.uint8(255 * heatmap)
    # Apply JET color mapping (dark spots = blue, important spots = bright red)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose heatmap on top of original BGR photograph mathematically
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    cv2.imwrite(str(out_path), superimposed_img)


def main() -> None:
    p = argparse.ArgumentParser(description="xAI Grad-CAM Generator for Thesis Explanations")
    p.add_argument("--image", type=Path, required=True, help="Path to raw farm photo")
    p.add_argument("--model", type=Path, required=True, help="Path to best_model.keras")
    p.add_argument("--architecture", choices=["cnn", "resnet50"], required=True)
    p.add_argument("--skip-preprocess", action="store_true", help="Skip OpenCV pipeline (use for Raw experiment)")
    p.add_argument("--out", type=Path, default=Path("gradcam_heatmap.png"), help="Output heatmap .png path")
    args = p.parse_args()

    print("[1] Loading Medical Model Architecture...")
    model = keras.models.load_model(args.model)
    # Remove softmax to ensure pure unbounded gradient flow
    model.layers[-1].activation = None 

    if args.architecture == "resnet50":
        last_conv_layer_name = "conv5_block3_out"
    else:
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if "conv" in layer.name.lower() and len(layer.output_shape) == 4:
                last_conv_layer_name = layer.name
                break
        if not last_conv_layer_name:
            raise SystemExit("Could not dynamically locate a Convolutional layer to run backpropagation!")
            
    print(f"    -> Isolated Target Feature Bank: '{last_conv_layer_name}'")

    print("[2] Processing Image...")
    img_array = get_img_array(args.image, args.skip_preprocess, size=224)
    if args.architecture == "resnet50":
        img_array = resnet50_preprocess_input(img_array)

    print("[3] Calculating Neural Gradients (Grad-CAM Mathematics)...")
    heatmap, pred_idx = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    print(f"[4] Generating Visual Overlay -> {args.out}")
    save_and_display_gradcam(args.image, heatmap, args.out)

    class_names_path = args.model.parent / "class_names.txt"
    if class_names_path.is_file():
        classes = class_names_path.read_text(encoding="utf-8").strip().split("\n")
        best_class = classes[pred_idx]
    else:
        best_class = f"ClassIndex_{pred_idx}"

    print("\n" + "="*50)
    print(" [XAI GRAD-CAM COMPLETE] ")
    print("="*50)
    print(f"  Underlying Prediction : {best_class}")
    print(f"  Visual exported to    : {args.out.resolve()}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
