# CNN-multiclass-Classification

Classfied the given images into 6 different classes - {0,1,2,3,4,5}.

# Dataset and preprocessing

Dataset was downloaded from Kaggle and image augmentation techniques were performed to expand the dataset.

# Model

The model was built using 3 convolutional layers and 2 dense layers at the end to predict the class of a given image.The callbacks used were EarlyStopping, ReduceLROnPlateau, ModelCheckpoint. The best weights are saved in '.h5' file and are loaded later in the model to give more accurate results.



