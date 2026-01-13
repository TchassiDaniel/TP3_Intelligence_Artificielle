# -*- coding: utf-8 -*-
"""
TP3 - Exercise 4: Neural Style Transfer.

This script applies the style of one image to the content of another using a
pre-trained VGG16 model.

@author: Tchassi Daniel
@matricule: 21P073
"""
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import time
import os

# --- Image Helper Functions ---

def load_and_process_image(image_path, target_size=(512, 512)):
    """Loads and preprocesses an image for VGG16."""
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
        
    img = img.resize(target_size)
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.vgg16.preprocess_input(img)
    return tf.convert_to_tensor(img)

def deprocess_image(processed_img):
    """Converts a tensor back to a displayable image."""
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)

    # VGG16-specific deprocessing
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  # BGR -> RGB

    x = np.clip(x, 0, 255).astype('uint8')
    return x

# --- VGG16 Feature Extractor ---

def get_vgg_extractor(style_layers, content_layers):
    """Builds a VGG16 model that returns features from specified layers."""
    vgg = keras.applications.VGG16(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in style_layers + content_layers]
    model = keras.Model(vgg.input, outputs)
    return model

# --- Loss Functions ---

def gram_matrix(input_tensor):
    """Calculates the Gram matrix for style representation."""
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

def style_loss_fn(style_outputs, style_targets, num_style_layers):
    """Computes the style loss."""
    loss = tf.add_n([
        tf.reduce_mean((gram_matrix(style_output) - gram_matrix(target))**2)
        for style_output, target in zip(style_outputs, style_targets)
    ])
    return loss / num_style_layers

def content_loss_fn(content_outputs, content_targets, num_content_layers):
    """Computes the content loss."""
    loss = tf.add_n([
        tf.reduce_mean((output - target)**2)
        for output, target in zip(content_outputs, content_targets)
    ])
    return loss / num_content_layers

# --- Main Training Step ---

@tf.function
def train_step(image, extractor, style_targets, content_targets, optimizer, weights, num_style_layers, num_content_layers):
    """Performs a single optimization step."""
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        s_outputs = outputs[:num_style_layers]
        c_outputs = outputs[num_style_layers:]

        s_loss = style_loss_fn(s_outputs, style_targets, num_style_layers)
        c_loss = content_loss_fn(c_outputs, content_targets, num_content_layers)
        total_loss = weights['style'] * s_loss + weights['content'] * c_loss

    gradients = tape.gradient(total_loss, image)
    optimizer.apply_gradients([(gradients, image)])
    image.assign(tf.clip_by_value(image, -150., 150.))

    return total_loss, s_loss, c_loss

# --- Main Execution Function ---

def run_exercise_4(content_path, style_path, epochs=10, steps_per_epoch=100):
    """Main function to run the style transfer."""
    mlflow.set_experiment("TP3-Exercise4-StyleTransfer")
    
    with mlflow.start_run(run_name="Ex4_StyleTransfer"):
        print("\nStarting Exercise 4: Neural Style Transfer...")

        # --- Parameters ---
        params = {
            "exercise": "4",
            "content_image": os.path.basename(content_path),
            "style_image": os.path.basename(style_path),
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "style_weight": 1e-2,
            "content_weight": 1e4,
            "learning_rate": 0.02
        }
        mlflow.log_params(params)
        print("Parameters logged to MLflow.")

        # --- Setup ---
        content_layers = ['block5_conv2']
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        num_content_layers = len(content_layers)
        num_style_layers = len(style_layers)
        
        extractor = get_vgg_extractor(style_layers, content_layers)
        
        content_image = load_and_process_image(content_path)
        style_image = load_and_process_image(style_path)
        
        if content_image is None or style_image is None:
            return # Exit if images failed to load

        # Extract target features
        style_targets = extractor(style_image)[:num_style_layers]
        content_targets = extractor(content_image)[num_style_layers:]

        generated_image = tf.Variable(content_image)
        
        optimizer = tf.optimizers.Adam(learning_rate=params['learning_rate'], beta_1=0.99, epsilon=1e-1)
        
        loss_weights = {
            'style': params['style_weight'],
            'content': params['content_weight']
        }
        
        output_dir = "style_transfer_results"
        os.makedirs(output_dir, exist_ok=True)
        mlflow.log_artifact(content_path, "inputs")
        mlflow.log_artifact(style_path, "inputs")

        # --- Optimization Loop ---
        start_time = time.time()
        print("\nStarting optimization loop...")
        for epoch in range(epochs):
            for step in range(steps_per_epoch):
                total_loss, s_loss, c_loss = train_step(
                    generated_image, extractor, style_targets, content_targets, optimizer, loss_weights, num_style_layers, num_content_layers
                )
            
            mlflow.log_metric("total_loss", total_loss.numpy(), step=epoch)
            mlflow.log_metric("style_loss", s_loss.numpy(), step=epoch)
            mlflow.log_metric("content_loss", c_loss.numpy(), step=epoch)
            
            print(f"Epoch {epoch + 1}/{epochs} | Total Loss: {total_loss.numpy():.2f}")

            if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
                img = deprocess_image(generated_image.numpy())
                img_path = os.path.join(output_dir, f"result_epoch_{epoch+1}.png")
                Image.fromarray(img).save(img_path)
                mlflow.log_artifact(img_path, "results")

        end_time = time.time()
        mlflow.log_metric("total_training_time_sec", end_time - start_time)

        final_image = deprocess_image(generated_image.numpy())
        final_path = os.path.join(output_dir, "final_result.png")
        Image.fromarray(final_image).save(final_path)
        mlflow.log_artifact(final_path, "results")
        
        print("\nExercise 4 finished successfully!")
        print(f"Results saved in '{output_dir}'. Check MLflow UI for detailed results.")

if __name__ == '__main__':
    # This allows running the script directly for testing
    content_dir = "images/content"
    style_dir = "images/style"
    if not (os.path.exists(content_dir) and os.listdir(content_dir)):
        os.makedirs(content_dir, exist_ok=True)
        print(f"'{content_dir}' is empty. Please add a content image there to run this script directly.")
    elif not (os.path.exists(style_dir) and os.listdir(style_dir)):
        os.makedirs(style_dir, exist_ok=True)
        print(f"'{style_dir}' is empty. Please add a style image there to run this script directly.")
    else:
        content_img_path = os.path.join(content_dir, os.listdir(content_dir)[0])
        style_img_path = os.path.join(style_dir, os.listdir(style_dir)[0])
        run_exercise_4(content_img_path, style_img_path)
