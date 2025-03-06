# 🧠 Implementing a Perceptron and Custom Neural Network on Japanese MNIST dataset.🧠
_Author: Thirada Tiamklang_

## ✔️ Overview
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

## 📊 Results

#### [Part A:](Assignment1-partA-14337188.ipynb) 
Forward and backpropagation implemented correctly.

#### [Part B:](Assignment1-partB-14337188.ipynb) 
Experiment 3, employing _the SGD optimizer with momentum=0.9, a learning
rate of 0.001, and weight decay=0.001, along with two hidden layers sized 256 and
128, and a dropout rate of 0.1_, yielded the most promising accuracy results at 90.09%.

In addition, The neural network architecture of each experiment is represented in the table below:
| Experiment | Optimizer | Learning Rate | Weight Decay | Input Size | Hidden Size       | Number of Classes | Dropout Rate | ReLU (Times) |
|------------|-----------|---------------|--------------|------------|-------------------|------------------|--------------|--------------|
| 1          | Adam      | 0.0001        | 0.001        | 784        | 128               | 10               | 0.5          | 1            |
| 2          | Adam      | 0.0001        | 0.001        | 784        | 128               | 10               | 0.1          | 1            |
| 3          | SGD (momentum=0.9) | 0.001 | 0.001        | 784        | 256,128           | 10               | 0.1          | 2            |
| 4          | SGD (momentum=0.9) | 0.001 | 0.001        | 784        | 512,256,128       | 10               | 0.1          | 3            |

Plots and model evaluations included in the notebook.

---
## 🤝 Acknowledgments

Thanks to the course instructors for providing the assignment template and guidance!
