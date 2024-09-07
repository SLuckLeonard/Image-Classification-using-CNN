from sklearn.model_selection import train_test_split


def preprocess_data(x_train, x_test, y_train, y_test, val_split=0.1):
    # Normalize the pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Split into validation data
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_split, random_state=42)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
