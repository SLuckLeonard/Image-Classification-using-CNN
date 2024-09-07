import matplotlib.pyplot as plt
import numpy as np

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def visualize_samples(x_train, y_train):
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    axes = axes.ravel()

    for i in np.arange(0, 10):
        axes[i].imshow(x_train[i])
        axes[i].set_title(class_names[int(y_train[i])])
        axes[i].axis('off')
    plt.subplots_adjust(hspace=0.5)
    plt.show()

def print_basic_stats(x_train, x_test):
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    print(f"Image shape: {x_train.shape[1:]}")
