import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

# Load the saved model
model = load_model('../saved_models/best_cnn_model.keras')

# Load CIFAR-10 data
from tensorflow.keras.datasets import cifar10
(_, _), (x_test, y_test) = cifar10.load_data()

# Preprocess test data (normalization)
x_test = x_test.astype('float32') / 255.0

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Make predictions
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = y_test.flatten()

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Classification Report
class_report = classification_report(y_true, y_pred_classes)
print("\nClassification Report:\n", class_report)

# Visualizing Confusion Matrix
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.show()

# Identify misclassifications
misclassified_indices = np.where(y_pred_classes != y_true)[0]
print(f"Number of misclassified images: {len(misclassified_indices)}")

# Plot some misclassified images
plt.figure(figsize=(12, 8))
for i, idx in enumerate(misclassified_indices[:9]):  # show 9 misclassifications
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[idx])
    plt.title(f"True: {y_true[idx]}, Pred: {y_pred_classes[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
