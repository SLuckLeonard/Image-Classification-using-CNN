import tensorflow as tf
from model.design_cnn import build_cnn_model

def compile_cnn_model(model):
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    model = build_cnn_model()
    model = compile_cnn_model(model)
    model.summary()
