
# Image Classification Project

## Overview
This project involves the development of a machine-learning model for image classification. The model is trained to classify images into 4 categories: Bicycles, Deers, Cars, Mountains. This README documents the development process, model design, training approach, and how to use the final product. 
I decided to use Convolutional Neural Networks (CNN) because they excel in image classification by effectively capturing spatial hierarchies. For example, AlexNet is a very accurate image classification CNN.

## Contents
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Design](#model-design)
- [Training the Model](#training-the-model)
- [Using the Model](#using-the-model)
- [Contributing](#contributing)

## Installation

If you want to train the model yourself start at step 1, if you want to use the website and make predictions start at step 4.

### Step 1 
Add the images in corresponding folders to train the ML model. The images folder has 4 folders: bicycles, cars, deers, mountains. Each of them has a txt file with the link to the database of images. Download the 4 databases and place them inside the folder.  


### Step 2

Start jyputer notebook and run all cells in the bdcm_convnn.ipynb file.

### Step 3 

Save the ML model.
```bash
from keras.models import load_model
model.save('model_name.keras')
```

### Step 4
```bash
# you have to be in the ..\image_classification_cnn_ml_model-main\bdcm_conv_nn_website directory
pip install requirements.txt
```
Load your model if you have made one. Pay attention to the file path. If you didn't train the model leave code as is.  
```bash
model = load_model('./model_name.keras')
```

### Step 5 
Run the flask app and make predictions.
```bash
# you have to be in the ..\image_classification_cnn_ml_model-main\bdcm_conv_nn_website directory
flask run
```

## Data Preprocessing
Preprocessing steps:
- Resizing images
- Normalization
- Data Augmentation
- Splitting the Dataset (only train and test (i do it with random), validation set is created from the fit module of keras)
- Label encoding

maybe explain why we need data preprocessing, and include examples, like why we need normalisation.

### Resizing images
I am resizing the images to 244*244 pixels. It is a reasonable compromise between having enough detail in the image for accurate classification and keeping the computational load manageable. For example, AlexNet uses 244 * 244. 

### Normalization
This step involves scaling pixel values to a standard range. I am dividing the image's pixel values by 255 to make it in the range of 0-1.

### Data Augmentation
To improve the robustness of the model and prevent overfitting, I will use the following data augmentation techniques: Rotation, Sharpening, Blurring, and Changing the contrast. It will be applied to 20% of all the images.

### Splitting the Dataset
I am creating a test dataset of 50 random samples to test the accuracy of the ML model. The validation dataset is created in the fit module of the ML model. As well as, the shuffling, it is set to true, it is used to reduce the chance of overfitting to specific sequences of data.

### Label encoding

LabelEncoder transforms y_train and y_test labels into integers. Then, to_categorical converts these integer labels into binary class matrices (one-hot encoding), necessary for multi-class classification tasks.

## Model Design

### Initial Basic Model 
**Accuracy: 46%**

- Started with a single dense layer. Choose 4 neurons to match the number of classes. 
- Used softmax for the output layer to obtain probabilities for each class, which is the most suitable for multi-class classification. 
- Selected Adam due to its adaptive learning rate properties, which is a good default choice for a wide range of problems, including image classification. 
- Choose the categorical cross-entropy loss function, which is the default loss function to use for multi-class classification problems.

```bash
# Initial Basic Model
model = models.Sequential([
    layers.Flatten(input_shape=input_shape),
    layers.Dense(4, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train_categorical, batch_size=64, epochs=10)
```

### Adding Convolutional Layers
**Accuracy: 86%**

- Added two Conv2D layers with max-pooling layers to learn the spatial features of the images. 
- Used 32 filters for both convolutional layers, as a starting point. 
- Relu activation was used in Conv2D layers due to its effectiveness in non-linear transformations. 
- 15% of the dataset is used as a validation set. 
- Shuffling the data each time an epoch begins to improve generalization and prevent order bias.

```bash
# Adding Convolutional Layers
model = models.Sequential([
    # input layer specified in the Conv2D layer
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(4, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train_categorical, batch_size=64, epochs=10, validation_split=0.15, shuffle=True)
```

### Adding Dropout Layers  
**Accuracy: 90%**

- Added dropout layers with a rate of 0.25 after max-pooling layers to reduce overfitting. This is done to create a more generalized model that performs better on unseen data.
- After adding two dropout layers to the model, I have seen a sudden decrease in accuracy from 86% to 78%. Due to this, I lowered the dropout (from 0.25 to 0.15) to retain more information. This had a positive effect on the accuracy from 78% to 90%.

```bash
# Adding Dropout Layers
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.15),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.15),
    layers.Flatten(),
    layers.Dense(4, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train_categorical, batch_size=64, epochs=10, validation_split=0.15, shuffle=True)
```

### Expanding the Network
**Accuracy: 96%**

- Added more convolutional layers, and increased the depth to 64 and to 128 neurons, to capture more complex features of the images. 
- Added the corresponding MaxPooling2D and Dropout layers, to reduce computational load, extract dominant features and encourage generalization.

```bash
# Expanding the Network
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.15),  
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.15),  
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dropout(0.15), 
    layers.Dense(4, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train_categorical, batch_size=64, epochs=10, validation_split=0.15, shuffle=True)
```

### Adjusting Parameters and experimenting with optimizers
**Accuracy: 94% - 98%**

- Added a dense layer with 128 neurons to allow the network to learn more complex functions from the high-level features extracted by the convolutional layers. 
- Adjusted the learning rate of the Adam optimizer to improve training stability and performance.
- This was a test that could possibly improve performance while creating an opportunity for more complex decision boundaries with the dense layer and increasing the stability of the learning rate.
- The accuracy dropped to 62%, therefore I decided to omit the second to last Dense layer with 128 neurons. 



```bash
# Adjusting Parameters
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.15),  
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.15),  
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dropout(0.15), 
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train_categorical, batch_size=64, epochs=10, validation_split=0.15, shuffle=True)
```

### Training the model 

- Implemented Data Augmentation to increase the diversity of the training dataset.
- Added about 6000 new samples to create a bigger dataset of data to train the model.
- I used the early stopping approach with 100 epochs and a batch size of 64, with the settings that if the validation loss does not improve by at least 0.001 for 10 consecutive epochs the training will stop.
- 50 randomly selected samples are used as a test dataset


### Final Model

```bash
# Final Model
model = models.Sequential([
    layers.Conv2D(12, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.05),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.05),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dropout(0.15),
    layers.Dense(4, activation='softmax')
])
model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10,  verbose=1, restore_best_weights=True )
model.fit(x_train, y_train_categorical, batch_size=64, epochs=100, validation_split=0.15, shuffle=True, callbacks=[early_stopping])
```

### Final Accuracy: 94% - 98%

## Contributing
Invite others to contribute and explain how they can do so. This might include:
- Reporting issues.
- Suggesting new features or improvements.
- Guidelines for submitting pull requests.

## Additional Notes
- Mention your process of adding more images to the dataset.
- Discuss challenges faced, such as dealing with images not conforming to labels, and how you addressed them.
- Reflect on the model's performance and potential areas for improvement.

