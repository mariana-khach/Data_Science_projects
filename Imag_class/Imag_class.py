# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 14:27:24 2024

@author: Mariana Khachatryan
"""


#200 images labeled as "lion" or "cheetah" 

import os
import random
import shutil

from PIL import Image

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img


#%%




lion_dir="images/Lions"
cheetah_dir="images/Cheetahs"

#Separate the data into training and validation samples
for data_path in [lion_dir,cheetah_dir]:
#data_path = "images/Lions"

    # path to destination folders
    train_folder = os.path.join(data_path, 'train')
    valid_folder = os.path.join(data_path, 'valid')
    
    # Define a list of image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Create a list of image filenames in 'data_path'
    imgs_list = [filename for filename in os.listdir(data_path) if os.path.splitext(filename)[-1] in image_extensions]
    
    # Sets the random seed 
    random.seed(42)
    
    # Shuffle the list of image filenames
    random.shuffle(imgs_list)
    
    # determine the number of images for each set
    train_size = int(len(imgs_list) * 0.7)
    valid_size = int(len(imgs_list) * 0.3)
    
    
    # Create destination folders if they don't exist
    for folder_path in [train_folder, valid_folder]:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    
    # Copy image files to destination folders
    for i, f in enumerate(imgs_list):
        if i < train_size:
            dest_folder = train_folder
        else:
            dest_folder = valid_folder
        shutil.copy(os.path.join(data_path, f), os.path.join(dest_folder, f))
        



#Directories of train and validation data
train_lions=lion_dir+"/train"
train_cheetahs=cheetah_dir+"/train"
valid_lions=lion_dir+"/valid"
valid_cheetahs=cheetah_dir+"/valid"


#%%

#We resize all the images to have 150 by 150 pixels, before feeding into neural net.
#We can also ommit this step since we can resize images inside flow_from_directory()

def resize_images(input_dir, output_dir, size=(150, 150)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(input_dir, filename))
            resized_img = img.resize(size)
            resized_img.save(os.path.join(output_dir, filename))


for input_directory in [train_lions,train_cheetahs,valid_lions,valid_cheetahs]:
    output_directory="images/"+str(input_directory).split("/")[-1]+"/"+str(input_directory).split("/")[-2]
    #output_directory=input_directory+"_resize"
    resize_images(input_directory, output_directory)
    


#%%
#Building convolutional neural network (CNN), with three modules, each containing
# convolution+relu(Rectified Linear Unit)+maxpooling(downsampling convolved feature)

#the input of the CNN is a 150*150*3 feature map, where 3 corresponds to red,green,blue 
input_img=layers.Input(shape=(150,150,3))

# 1st convolution extracts 16 filters that are 3x3
# and is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(16, 3, activation='relu')(input_img)
x = layers.MaxPooling2D(2)(x)

# 2nd convolution extracts 32 filters that are 3x3
# and is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# 3rd convolution extracts 64 filters that are 3x3
# and is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)


#Flattening feature map and adding two fully-connected layers
x = layers.Flatten()(x)

# layer with ReLU activation and 512 hidden units
x = layers.Dense(512, activation='relu')(x)

#layer with a single node and sigmoid activation for classification
output = layers.Dense(1, activation='sigmoid')(x)

# Our model takes as an input "input_img" feature map 
#and returns input feature map + stacked convolution/maxpooling layers + fully 
# connected layer + sigmoid output layer
model = Model(input_img, output)




#%%
#Compiling our model
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['acc'])

#%%

train_dir="images/train"
valid_dir="images/valid"

#Defining data generators to read the pictures and store in float32 tensors and 
#normalizing pixel value range [0,255] to [0,1]
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 14 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=14,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow validation images in batches of 6 using val_datagen generator
valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(150, 150),
        batch_size=6,
        class_mode='binary')


#%%
#Training the model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=10, # 140 images = batch_size * steps
      epochs=15,
      validation_data=valid_generator,
      validation_steps=10,  # 60 images = batch_size * steps
      verbose=2)


#%%

#Evaluating accuracy and loss

# Retrieve a list of accuracy results on training and validation data
# sets for each training epoch
acc = history.history['acc']
val_acc = history.history['val_acc']

# Retrieve a list of list results on training and validation data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation  loss')



#It looks like we have such a small dataset that we are overfitting and that is 
#why we see an improvement in accuracy and loss for training sample but not for validation

#%%

#to prevent overfitting we add augmentation  (image rotation, zomming etc. ) 
#to preprocessing so that our model will never see the exact same picture
# during training

# Adding rescale, rotation_range, width_shift_range, height_shift_range,
# shear_range, zoom_range, and horizontal flip to our ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

# Leaving validation data unchanged
val_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 32 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=10,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow validation images in batches of 32 using val_datagen generator
validation_generator = val_datagen.flow_from_directory(
        valid_dir,
        target_size=(150, 150),
        batch_size=10,
        class_mode='binary')


#%%

# We also add random dropout of some fraction of layers in each training step


#Building convolutional neural network (CNN), with three modules, each containing
# convolution+relu+maxpooling

#the input of the CNN is a 150*150*3 feature map, where 3 corresponds to red,green,blue 
input_img=layers.Input(shape=(150,150,3))

# 1st convolution extracts 16 filters that are 3x3
# and is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(16, 3, activation='relu')(input_img)
x = layers.MaxPooling2D(2)(x)

# 2nd convolution extracts 32 filters that are 3x3
# and is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# 3rd convolution extracts 64 filters that are 3x3
# and is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)


#Flattening feature map and adding two fully-connected layers
x = layers.Flatten()(x)

# layer with ReLU activation and 512 hidden units
x = layers.Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

#layer with a single node and sigmoid activation for classification
output = layers.Dense(1, activation='sigmoid')(x)

# Our model takes as an input "input_img" feature map 
#and returns input feature map + stacked convolution/maxpooling layers + fully 
# connected layer + sigmoid output layer
model = Model(input_img, output)
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['acc'])


#Train new model

history = model.fit_generator(
      train_generator,
      steps_per_epoch=14,      # 140 images = batch_size * steps
      epochs=20,
      validation_data=validation_generator,
      validation_steps=6,     # 60 images = batch_size * steps
      verbose=2)


#%%

#Evaluating accuracy and loss

# Retrieve a list of accuracy results on training and validation data
# sets for each training epoch
acc = history.history['acc']
val_acc = history.history['val_acc']

# Retrieve a list of list results on training and validation data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation  loss')


#We can see improvement in the performance of our model
#The training and validation accuracy increase and the loss for both decreases


