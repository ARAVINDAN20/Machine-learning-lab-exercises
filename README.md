# Machine-learning-lab-exercises

Welcome to the **Machine Learning Lab Exercises** repository! This repository contains a collection of practical experiments that cover various concepts, techniques, and algorithms in machine learning. Each experiment is implemented in Python using popular libraries like Pandas, Scikit-learn, Matplotlib, TensorFlow, and PyTorch. Below is the detailed documentation of all experiments included in this repository.

---

## Table of Contents

1. [Data Preprocessing Techniques](#1-data-preprocessing-techniques)
2. [Find-S Model for Concept Learning](#2-find-s-model-for-concept-learning)
3. [Univariate Plots for Data Exploration](#3-univariate-plots-for-data-exploration)
4. [Principal Component Analysis (PCA)](#4-principal-component-analysis-pca)
5. [Single-layer Neural Network](#5-single-layer-neural-network)
6. [Backpropagation for Training MLP](#6-backpropagation-for-training-mlp)
7. [Support Vector Machines (SVM)](#7-support-vector-machines-svm)
8. [AdaBoost Algorithm](#8-adaboost-algorithm)
9. [Random Forest Algorithm](#9-random-forest-algorithm)
10. [Gaussian Mixture Models (GMM)](#10-gaussian-mixture-models-gmm)
11. [Neural Network Applications](#11-neural-network-applications)
12. [Time Series Forecasting with ARIMA](#12-time-series-forecasting-with-arima)
13. [How to Use](#how-to-use)
14. [License](#license)
15. [Author](#author)

---

### 1. Data Preprocessing Techniques
**Objective**: Learn to preprocess datasets using Pandas and Scikit-learn.

**Key Topics**:
- Handling missing data
- Encoding categorical variables
- Feature scaling and normalization

**Libraries Used**: Pandas, Scikit-learn

---

### 2. Find-S Model for Concept Learning
**Objective**: Implement the Find-S algorithm to identify the most specific hypothesis from training data.

**Key Topics**:
- Concept learning
- Hypothesis space

**Libraries Used**: Scikit-learn

---

### 3. Univariate Plots for Data Exploration
**Objective**: Explore datasets using univariate visualizations.

**Key Topics**:
- Histograms
- Boxplots
- KDE plots

**Libraries Used**: Matplotlib, Seaborn

---

### 4. Principal Component Analysis (PCA)
**Objective**: Understand the concept of PCA and its applications in dimensionality reduction.

**Key Topics**:
- Variance explained by components
- Visualizing reduced dimensions

**Libraries Used**: Scikit-learn, Matplotlib

---

### 5. Single-layer Neural Network
**Objective**: Implement a single-layer neural network for binary classification.

**Key Topics**:
- Perceptron model
- Gradient descent

**Libraries Used**: TensorFlow

---

### 6. Backpropagation for Training MLP
**Objective**: Train a Multilayer Perceptron (MLP) using the backpropagation algorithm.

**Key Topics**:
- Feedforward neural networks
- Loss function optimization

**Libraries Used**: Scikit-learn, TensorFlow

---

### 7. Support Vector Machines (SVM)
**Objective**: Apply SVM for classification and regression tasks.

**Key Topics**:
- Kernel functions
- Hyperplane visualization

**Libraries Used**: Scikit-learn

---

### 8. AdaBoost Algorithm
**Objective**: Implement the AdaBoost algorithm for boosting ensemble techniques.

**Key Topics**:
- Weak learners
- Weighted ensemble

**Libraries Used**: Scikit-learn

---

### 9. Random Forest Algorithm
**Objective**: Build a Random Forest model for classification and regression.

**Key Topics**:
- Bagging
- Feature importance

**Libraries Used**: Scikit-learn

---

### 10. Gaussian Mixture Models (GMM)
**Objective**: Use Gaussian Mixture Models for clustering.

**Key Topics**:
- Expectation-Maximization algorithm
- Clustering evaluation

**Libraries Used**: Scikit-learn, Matplotlib

---

### 11. Neural Network Applications
**Objective**: Explore real-world applications of neural networks.

**Key Topics**:
- Image recognition
- Natural language processing

**Libraries Used**: TensorFlow, PyTorch

---

### 12. Time Series Forecasting with ARIMA
**Objective**: Forecast time series data using ARIMA.

**Key Topics**:
- Stationarity testing
- Autoregressive models

**Libraries Used**: Statsmodels, Matplotlib

---

### 13. How to Use

#### Using Visual Studio Code

1. Open Visual Studio Code and navigate to the folder containing this repository.
2. Install the Python extension for VS Code if not already installed.
3. Open a terminal within VS Code and create a virtual environment:
   ```bash
   python -m venv ml_env
   ml_env\Scripts\activate  # On Windows
   source ml_env/bin/activate  # On macOS/Linux
   ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Open any `.ipynb` file and use the Jupyter Notebook extension in VS Code to run the experiments.

#### Using Google Colab

1. Go to [Google Colab](https://colab.research.google.com/).
2. Upload the desired `.ipynb` file by clicking on **File > Upload Notebook**.
3. Ensure all required libraries are installed in the Colab environment by running:
   ```python
   !pip install numpy pandas scikit-learn matplotlib seaborn tensorflow torch statsmodels
   ```
4. Run the cells sequentially to execute the experiments.

#### Running Locally

1. Clone the repository to your local system:
   ```bash
   git clone https://github.com/ARAVINDAN20/Machine-learning-lab-exercises.git
   ```
2. Navigate to the repository directory:
   ```bash
   cd Machine-learning-lab-exercises
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv ml_env
   ml_env\Scripts\activate  # On Windows
   source ml_env/bin/activate  # On macOS/Linux
   pip install -r requirements.txt
   ```
4. Open the Jupyter Notebook server:
   ```bash
   jupyter notebook
   ```
5. Run the `.ipynb` files in your browser.

---

### 14. License
This repository is licensed under the MIT License.

---

### 15. Author
[Aravindan](https://github.com/ARAVINDAN20)

