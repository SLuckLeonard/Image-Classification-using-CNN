# main.py
from data.download_data import download_cifar10
from data.preprocess_data import preprocess_data
from visualization.visualize_data import visualize_samples, print_basic_stats
from augment.augment_data import get_data_augmentation
from model.design_cnn import build_cnn_model
from model.compile_model import compile_cnn_model
import tensorflow as tf
import os

# Define a directory to save the model
MODEL_DIR = "saved_models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_cnn_model.keras")

if __name__ == "__main__":
    # Step 1: Download the data
    (x_train, y_train), (x_test, y_test) = download_cifar10()

    # Step 2: Preprocess the data (normalize and split)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocess_data(x_train, x_test, y_train, y_test)

    # Step 3: Visualize sample data and print statistics
    visualize_samples(x_train, y_train)
    print_basic_stats(x_train, x_test)

    # Step 4: Augment the data
    datagen = get_data_augmentation()

    # Step 5: Build and compile the model
    model = build_cnn_model()
    model = compile_cnn_model(model)

    # Step 6: Train the model and track metrics
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        epochs=30,
        validation_data=(x_val, y_val),
        verbose=1
    )

    # Step 7: Save the best model manually
    # Create the directory if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Save the model in the newer Keras format
    model.save(MODEL_PATH)
    print(f"Model saved at: {MODEL_PATH}")

    # Evaluate the model on the test data
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")
