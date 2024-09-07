from data.download_data import download_cifar10
from data.preprocess_data import preprocess_data
from visualization.visualize_data import visualize_samples, print_basic_stats
from augment.augment_data import augment_data

if __name__ == "__main__":
    # Step 1: Download the data
    (x_train, y_train), (x_test, y_test) = download_cifar10()

    # Step 2: Preprocess the data (normalize and split)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocess_data(x_train, x_test, y_train, y_test)

    # Step 3: Visualize sample data and print statistics
    visualize_samples(x_train, y_train)
    print_basic_stats(x_train, x_test)

    # Step 4: Augment the data (if needed)
    datagen = augment_data(x_train)

    print("Data ready for model training and augmentation.")
