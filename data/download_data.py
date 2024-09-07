import tensorflow as tf
from tensorflow.keras.datasets import cifar10

def download_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return (x_train, y_train), (x_test, y_test)
