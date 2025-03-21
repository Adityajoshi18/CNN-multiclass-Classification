<!-- # CNN-multiclass-Classification

Classfied the given images into 6 different classes - {0,1,2,3,4,5}.

# Dataset and preprocessing

Dataset was downloaded from Kaggle and image augmentation techniques were performed to expand the size of the dataset.

# Model

The model was built using 3 convolutional layers and 2 dense layers at the end to predict the class of a given image.The callbacks used were EarlyStopping, ReduceLROnPlateau, ModelCheckpoint. The best weights are saved in '.h5' file and are loaded later in the model to give more accurate results. -->

# CNN-multiclass-Classification

<!-- Developed a Siamese Deep Network called MaLSTM (Manhattan LSTM) to detect similarity between two sentences. -->

## Overview

This project implements a Convolutional Neural Network (CNN) model for classifying images into six different classes: {0, 1, 2, 3, 4, 5}. The model has been trained using an augmented dataset to improve accuracy and generalization.

## Dataset and Preprocessing

- The dataset was obtained from Kaggle.
- Image augmentation techniques were applied to increase the dataset size and improve model performance.
- The dataset is structured as follows:
    - train_set.csv: Contains the training data labels.
    - sample_submission.csv: Provides a template for model predictions.

## Model Architecture

- The CNN model consists of:
    - **Three Convolutional Layers**
    - **Two Fully Connected (Dense) Layers** at the end to predict the class label.
    - **Activation Functions**: ReLU for convolutional layers and Softmax for the output layer.
<!-- - Image augmentation techniques were applied to increase the dataset size and improve model performance.
- The dataset is structured as follows:
    - train_set.csv: Contains the training data labels.
    - sample_submission.csv: Provides a template for model predictions. -->

## Callbacks Used

- **EarlyStopping**: Stops training when the model performance stops improving.
- **ReduceLROnPlateau**: Reduces the learning rate when the validation loss stops decreasing.
- **ModelCheckpoint**: Saves the best model weights in a .h5 file.

## Files in the Project

- CNN classification.ipynb: Jupyter Notebook containing the full implementation of the model.
- train_set.csv: Training dataset with labels.
- sample_submission.csv: Sample file for submission format.
- .h5 file: Stores the best weights of the trained model.

## How to Run

1. Install the required dependencies:

```bash
pip install tensorflow keras numpy pandas matplotlib
```
2. Open the Jupyter Notebook and run all cells.
3. Load the saved weights for inference to get more accurate predictions.

## Results

The trained CNN model achieves improved accuracy due to dataset augmentation and optimized hyperparameters. Predictions can be generated using the trained model weights.

## Future Improvements

- Experiment with deeper architectures for better accuracy.
- Use Transfer Learning to leverage pre-trained models.
- Implement real-time image classification.





