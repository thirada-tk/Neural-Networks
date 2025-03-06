# 🧠 Implementing a Perceptron and Custom Neural Network on Japanese MNIST dataset.
----------------------------------------------------------------------------------
## Overview
This project is divided into two parts:
1. Part A: Perceptron from Scratch (Using NumPy)
* Implement a simple perceptron model from scratch.
* Perform forward and backpropagation on a small dataset with 3 features.
* Allowed libraries: NumPy, Pandas (No Scikit-learn or TensorFlow).
* Predefined template provided with TODO sections to complete.
2. Part B: Custom Neural Network with PyTorch
* Train a fully connected neural network on the Japanese MNIST dataset.
* Dataset: __70,000 images of handwritten Hiragana characters.__
* Input size: __Flattened 28x28 images → 784 features.__
* Target: __10 different character classes.__
* Goal: Train a model with __at least 80% accuracy__ while minimizing overfitting.
* Allowed layers: __Fully connected, Dropout (No convolution layers).__
* Some code is provided, with TODO sections to complete.

 ## 📁 Dataset

* __Part A__: Small dataset with 3 features.
* __Part B__: Japanese MNIST dataset (Hiragana characters, 70,000 images).

## 🔧 Installation

To set up the environment, install dependencies:
```python
pip install numpy pandas torch torchvision matplotlib
```

## 🚀 Usage

#### Part A (Perceptron with NumPy)

Open notebooks/perceptron_numpy.ipynb in Jupyter Notebook.

Implement missing parts marked with TODO.

Run the notebook to train the perceptron model.

#### Part B (Neural Network with PyTorch)

Open notebooks/neural_net_pytorch.ipynb.

Implement missing parts (marked with TODO).

Run experiments to achieve 80% accuracy with minimal overfitting.

📊 Results

Part A: Forward and backpropagation implemented correctly.

Part B: Final accuracy and loss metrics.

Plots and model evaluations included in the notebook.

📌 Notes

Assertions are included to validate outputs. If no errors occur, the implementation is correct.

Ensure models are not overfitting while achieving the required accuracy.

📖 References

Japanese MNIST Dataset: Dataset Link (if available)

🤝 Acknowledgments

Thanks to the course instructors for providing the assignment template and guidance!
