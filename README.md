# Advanced-Machine-Learning-Pattern-Recognition-Suite
This repository features a robust collection of machine learning algorithms built entirely from scratch, alongside a comprehensive classification pipeline for high-dimensional data. The project demonstrates a deep theoretical understanding of statistical modeling and the practical engineering required to deploy state-of-the-art predictive models.

## 🚀 Core Architecture & Modules
### 1. Custom Maximum Likelihood Estimation (MLE) Engine
- Objective: Estimate the core parameters (Mean and Covariance) of distinct 2D normal distributions representing different classes.

- Engineering: Built a custom MLE calculator using pure mathematical matrix operations, deliberately avoiding high-level statistical library functions (like np.mean or np.cov).

- Impact: Successfully extracted unbiased estimators for multi-class data. Developed a custom bivariate normal PDF algorithm to render the resulting probability density functions in a unified, color-coded 3D surface plot.

### 2. Non-Parametric Density Estimation (Parzen Windows)
- Objective: Perform non-parametric density estimation to model complex data distributions.

- Engineering: Developed a Parzen Windows estimator from the ground up, integrating both Hypercube (Uniform) and Gaussian kernel functions.

- Impact: Engineered a hyperparameter tuning loop to isolate the optimal bandwidth (h) that minimizes the Mean Squared Error (MSE) against theoretical baselines. Successfully demonstrated the bias-variance tradeoff and the superior smoothing capabilities of Gaussian kernels.

### 3. K-Nearest Neighbors (KNN) from Scratch
- Objective: Build a complete KNN classification system without relying on pre-packaged ML libraries for the core logic.

- Engineering: Manually coded the foundational mathematical components, including optimized Euclidean distance computation, neighbor sorting algorithms, and probabilistic class prediction.

- Impact: Automated the evaluation of the k hyperparameter to balance variance and bias, achieving a peak accuracy of 76.0% (at k=16). Generated rich, meshgrid-based visualizations using contourf to map non-linear decision boundaries and class transitions across the feature space.

### 4. High-Dimensional Classification Pipeline
- Objective: Design and deploy a production-ready machine learning pipeline to classify a massive, high-dimensional dataset (8,743 training samples, 224 features, 5 distinct classes).

- Data Preprocessing: Implemented a robust standardization step (StandardScaler) followed by Principal Component Analysis (PCA). By retaining 95% of the variance, the feature space was compressed from 224 to 190 components, drastically improving computational efficiency and filtering out noise.

- Model Selection & Tuning: Benchmarked a diverse suite of state-of-the-art models including Random Forest, Support Vector Machines (SVM with RBF kernel), Logistic Regression, KNN, and XGBoost using robust k-fold cross-validation.

- Impact: The SVM (RBF) classifier emerged as the top performer, achieving an 86.16% accuracy on the hold-out validation set. Handled complex overlapping feature distributions and exported highly accurate predictions for a blind test set of 6,955 samples.

## 🛠️ Technologies & Stack
- Python 3

- NumPy: Core matrix operations, linear algebra, and from-scratch algorithm construction.

- Pandas: Data manipulation and ingestion.

- Matplotlib: Advanced 2D/3D visualizations, histograms, and decision boundary mapping.

- Scikit-Learn: PCA, SVM, Random Forest, Cross-Validation, and performance metrics.

## 👨‍💻 Developed By
- Eleni Lazaridou [https://github.com/Elena440Hz]
- Alexandros Souroullas [https://github.com/asouroul]
