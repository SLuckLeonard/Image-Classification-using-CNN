from data.download_data import download_cifar10
from data.preprocess_data import preprocess_data
from visualization.visualize_data import visualize_samples, print_basic_stats
from augment.augment_data import get_data_augmentation
from model.design_cnn import build_cnn_model
from model.compile_model import compile_cnn_model
import tensorflow as tf

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

    # Step 6: Train the model
    model.fit(datagen.flow(x_train, y_train, batch_size=64),
              epochs=30,
              validation_data=(x_val, y_val),
              verbose=1)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")

    print("Model training complete.")
