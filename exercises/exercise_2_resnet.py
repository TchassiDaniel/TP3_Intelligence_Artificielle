# -*- coding: utf-8 -*-
"""
TP3 - Exercise 2: ResNet for Image Classification.

This script builds, trains, and evaluates a ResNet model on the CIFAR-10 dataset.
Results are tracked using MLflow.

@author: Tchassi Daniel
@matricule: 21P073
"""
import mlflow
import mlflow.tensorflow
from tensorflow import keras
from sklearn.metrics import classification_report
import numpy as np

from src.data_loader import load_cifar10_data
from src.models import build_resnet

def run_exercise_2():
    """
    Main function to run the entire Exercise 2 pipeline.
    """
    # Define the MLflow experiment
    mlflow.set_experiment("TP3-Exercise2-ResNet")

    # Load and preprocess data
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = load_cifar10_data()

    # Start an MLflow run
    with mlflow.start_run(run_name="Ex2_ResNet_CIFAR10"):
        print("\nStarting Exercise 2: ResNet Training...")

        # --- MLflow Tracking: Parameters ---
        params = {
            "exercise": "2",
            "model_type": "resnet",
            "dataset": "cifar10",
            "epochs": 15,
            "batch_size": 64,
            "optimizer": "adam",
            "loss_function": "categorical_crossentropy",
            "residual_blocks": 3,
            "dropout_rate": 0.5
        }
        mlflow.log_params(params)
        print("Parameters logged to MLflow.")

        # --- Model Building ---
        model = build_resnet(input_shape, num_classes)
        model.compile(
            optimizer=params["optimizer"],
            loss=params["loss_function"],
            metrics=['accuracy']
        )
        # Log model summary as a text artifact
        summary_lines = []
        model.summary(print_fn=lambda x: summary_lines.append(x))
        summary = "\n".join(summary_lines)
        mlflow.log_text(summary, "model_summary.txt")
        print("Model architecture built and summary logged.")

        # --- Callbacks ---
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )

        # --- Model Training ---
        print("\nTraining the model...")
        history = model.fit(
            x_train, y_train,
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            validation_split=0.1,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        print("Training complete.")

        # --- MLflow Tracking: Metrics ---
        # Log training history
        for epoch in range(len(history.history['loss'])):
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
        print("Training and validation metrics logged per epoch.")

        # --- Model Evaluation ---
        print("\nEvaluating the model on the test set...")
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        # --- MLflow Tracking: Final Metrics and Report ---
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)

        # Generate and log classification report
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        report_dict = classification_report(y_true_classes, y_pred_classes, output_dict=True)
        report_str = classification_report(y_true_classes, y_pred_classes, digits=4)

        mlflow.log_metric("precision_weighted", report_dict['weighted avg']['precision'])
        mlflow.log_metric("recall_weighted", report_dict['weighted avg']['recall'])
        mlflow.log_metric("f1_score_weighted", report_dict['weighted avg']['f1-score'])

        mlflow.log_text(report_str, "classification_report.txt")
        print("Final metrics and classification report logged.")
        
        # --- MLflow Tracking: Model Artifact ---
        mlflow.tensorflow.log_model(model, artifact_path="model")
        print("Model logged as an MLflow artifact.")

        print("\nExercise 2 finished successfully!")
        print("Check the MLflow UI for detailed results.")

if __name__ == '__main__':
    run_exercise_2()
