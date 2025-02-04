from keras.datasets import mnist
import numpy as np

@st.cache_data
def get_mnist_data():
    return mnist.load_data()


data = get_mnist_data()
print(data)
