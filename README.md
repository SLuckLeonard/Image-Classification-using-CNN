# Image-Classification-using-CNN
This repository contains an end-to-end project for image classification using Convolutional Neural Networks (CNN) on the CIFAR-10 dataset. It demonstrates data acquisition, preprocessing, model development, training, and saving a deep learning model in TensorFlow/Keras.

## Project Overview
The goal of this project is to create a CNN model capable of classifying images from the CIFAR-10 dataset, a widely-used dataset consisting of 60,000 32x32 color images in 10 different classes. This project is structured to be modular and scalable, allowing for easy extension and experimentation.

## Progress So Far
<b>Data Acquisition:</b> We have successfully downloaded and visualized the CIFAR-10 dataset.
<b>Data Preprocessing:</b> The dataset has been normalized and split into training, validation, and test sets.
<b>Data Augmentation:</b> Basic data augmentation techniques (rotation, zoom, flipping) have been applied to improve model generalization.
<b>CNN Model Architecture:</b> A basic CNN architecture has been designed and compiled with appropriate loss function and optimizer.
<b>Model Training:</b> The CNN has been trained for 10 epochs with tracking of accuracy and loss, and the model has been saved in the .keras format.
The current version of the project saves the trained model and logs accuracy and loss during training.

## How to Use this Project
Requirements
Make sure you have the following dependencies installed:

1. Python 3.8 or higher
2. TensorFlow 2.x
3. NumPy
4. Matplotlib
You can install all dependencies using the following command:

```bash
pip install -r requirements.txt
```
##Running the Project
- Clone the Repository:

```bash
git clone https://github.com/your-username/Image-Classification-using-CNN.git
cd Image-Classification-using-CNN
```
- Download CIFAR-10 Dataset:

The CIFAR-10 dataset will automatically be downloaded when running the script.

- Run the Main Script:

To train the model and save it, simply run:

```bash
python main.py
```
This will:

Download the CIFAR-10 dataset.
Preprocess the data (normalize and split).
Visualize the dataset and print basic statistics.
Apply data augmentation.
Train the CNN model for 10 epochs.
Save the model in the .keras format at saved_models/best_cnn_model.keras.
Log and print training accuracy and loss.
- Check the Saved Model:

The saved model can be found in the saved_models folder as best_cnn_model.keras.

## Project Structure
The project has a clean, modular structure for better code organization and scalability:

```bash
Image-Classification-using-CNN/
│
├── augment/                     # Data augmentation functions
│   └── augment_data.py
│
├── data/                        # Data acquisition and preprocessing
│   ├── download_data.py
│   ├── preprocess_data.py
│   └── __init__.py
│
├── model/                       # Model architecture and compilation
│   ├── design_cnn.py
│   ├── compile_model.py
│   └── __init__.py
│
├── visualization/               # Visualization of data samples
│   ├── visualize_data.py
│   └── __init__.py
│
├── saved_models/                # Directory for saving the trained model
│   └── best_cnn_model.keras
│
├── main.py                      # Main script for training and saving the model
├── README.md                    # This file
└── requirements.txt             # List of dependencies
```
## Next Steps
Add a more complex CNN architecture to improve performance.
Implement advanced techniques like learning rate scheduling or early stopping.
Explore model evaluation metrics and performance analytics.
Contributing
Feel free to fork the repository, create a new branch, and submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
