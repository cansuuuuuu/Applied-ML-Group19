import os
import tensorflow as tf
from project_name.data.loading.loader import load_dataset
from train_cnn import prepare_for_cnn
from project_name.models.cnn import CNNModel


def main():
    model_path = "project_name/models/cnn_model.keras"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return
    
    model = tf.keras.models.load_model(model_path)
    model.summary()

    cnn_wrapper = CNNModel(input_shape=model.input_shape[1:], num_classes=model.output_shape[1])
    cnn_wrapper.model = model

    X_train, y_train, X_test, y_test, classes = load_dataset()
    
    idx = 0
    raw_img = X_test[idx]
    label = y_test[idx]
    
    print(f"Selected image of class: {classes[label]}")

    img_array = prepare_for_cnn([raw_img])
    
    heatmap = cnn_wrapper.generate_gradcam(img_array)

    output_path = "gradcam_result.png"
    cnn_wrapper.save_gradcam(raw_img, heatmap, save_path=output_path)
    
    print(f"Grad-CAM result saved to {output_path}")

if __name__ == "__main__":
    main()
