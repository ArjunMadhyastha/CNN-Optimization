# CNN-Optimization for Fashion MNIST Classification
This project implements and optimizes a Convolutional Neural Network (CNN) for image classification on the Fashion MNIST dataset. The model's hyperparameters are fine-tuned using Keras Tuner, allowing for automatic discovery of the best configuration for improved accuracy and performance.
  # Features
  * Custom CNN Architecture:  The model consists of two convolutional layers, followed by a dense layer for image classification.
  * Hyperparameter Tuning: Utilizes Keras Tuner's RandomSearch to optimize key 
    hyperparameters such as:
    - Number of filters in each convolutional layer.
    - Kernel sizes for convolution.
    - Units in the dense layer.
    - Learning rates for the Adam optimizer.
  * Regularization:  L2 regularization is applied to prevent overfitting.
  * Data Preprocessing: The Fashion MNIST dataset is normalized and reshaped 
    to prepare it for CNN input.
  # Dataset
  * The project uses the Fashion MNIST dataset, which contains 70,000 
    grayscale images of 10 different clothing categories. The dataset is 
    already included in TensorFlow/Keras, and it is divided into 60,000 
    training images and 10,000 test images.
  # Key Steps
  * Model Architecture:
    - Two convolutional layers with ReLU activation.
    - A flattening layer to convert 2D outputs to a 1D vector.
    - A dense layer with a softmax output for multi-class classification (10 
      classes).
  * Hyperparameter Tuning:
    - Keras Tuner is used to perform a random search across different 
      hyperparameter combinations:
        - Number of filters in the convolutional layers.
        - Kernel sizes for convolution.
        - Units in the dense layer.
        - Learning rates for the Adam optimizer.
  * Training & Evaluation:
    - The model is trained on 90% of the training data, with 10% reserved 
      for validation.
    - The best model configuration is selected after running multiple trials.
    - Performance is evaluated on the test dataset using metrics like 
      accuracy.
  # Hyperparameters Tuned
  * The following hyperparameters are optimized using Keras Tuner:
    - Number of filters in the convolution layers (ranging from 32 to 128).
    - Kernel sizes (3x3 and 5x5).
    - Dense layer units (ranging from 32 to 128).
    - Learning rate (1e-2 or 1e-3).
  # Requirements
  * TensorFlow
  * Keras
  * Numpy
  * Matplotlib
  * Scikit-learn
  * Keras Tuner
  You can install the necessary packages using the following command:
- pip install -r requirements.txt
 # How to Run
 * Clone the respository:
   - git clone <repo-link>
 * Install the required dependencies.
 * Run the Jupyter notebook CNN_Optimization.ipynb to begin the 
   hyperparameter tuning process.
 * The model will be trained and evaluated, and the best model configuration 
   will be displayed after tuning.
# Results
* Accuracy: The optimized model achieved an accuracy of around 91% on the 
  test dataset.
* Tuning Performance: Keras Tuner successfully identified the best 
  hyperparameter combination for maximizing model accuracy.
# Conclusion
This project demonstrates the power of hyperparameter tuning for CNN optimization, yielding a model with improved performance on the Fashion MNIST dataset.
