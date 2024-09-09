# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:42:07 2024

@author: Mariana Khachatryan
"""

from PIL import Image
import os
import random
import shutil
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import seaborn as sns
import numpy as np



def calculate_average_pixel_size(directory):
    total_width = 0
    total_height = 0
    num_images = 0

    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter for image files
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    # Iterate over image files
    for image_file in image_files:
        try:
            # Open image using PIL
            with Image.open(os.path.join(directory, image_file)) as img:
                width, height = img.size
                total_width += width
                total_height += height
                num_images += 1
        except IOError:
            # Handle IOError if image file cannot be opened
            print(f"Skipping {image_file} due to IOError")

    # Calculate average pixel size
    if num_images > 0:
        average_width = total_width / num_images
        average_height = total_height / num_images
        print(f"Average image size: {average_width} x {average_height} pixels")
    else:
        print("No images found in the directory")



def train_test_data_sep(class1,class2,imdir,bigdir_name="train",smalldir_name="valid",test_fraction=0.3):
    


    #Separate the data into training and validation samples
    for class_nm in [class1,class2]:
        data_path=imdir+"/"+class_nm
        #data_path = "images/Lions"
        # path to destination folders
        train_folder =imdir+"/"+bigdir_name+"/"+class_nm
        valid_folder = imdir+"/"+smalldir_name+"/"+class_nm



        # Define a list of image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Create a list of image filenames in 'data_path'
        imgs_list = [filename for filename in os.listdir(data_path) if os.path.splitext(filename)[-1] in image_extensions]
        
        # Sets the random seed 
        random.seed(42)
        
        # Shuffle the list of image filenames
        random.shuffle(imgs_list)
        
        # determine the number of images for each set
        train_size = int(len(imgs_list) * (1-test_fraction))
        valid_size = int(len(imgs_list) * test_fraction)
        
        
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




class model_eval:
    
    def __init__(self,model,test_image_gen):
        
        self.model=model
        self.test_image_gen=test_image_gen
        
        
    def model_pred(self):
        
        pred_class=(self.model.predict(self.test_image_gen)>0.5).astype(int)
        true_class=self.test_image_gen.classes
        print(classification_report(true_class,pred_class))
        ConfusionMatrixDisplay.from_predictions(true_class, pred_class)
        
        
    def loss_plot(self,losses):
        
        
        print("Plotting image for losses"+str(losses.columns))
        sns.lineplot(data=losses,markers=True)

        plt.show()




    def pred_img_class(self,img_path,image_shape):
        
        #We need the picture full path and the pixelsize of images that model has been trained with
        my_image=image.load_img(img_path,target_size=image_shape)
        my_image_arr=image.img_to_array(my_image)
        plt.imshow(my_image_arr.astype('uint8'))
        my_image_arr=np.expand_dims(my_image_arr,axis=0)#because model is trained with batch of images
        predclass=self.model.predict(my_image_arr)
        class_dic=self.test_image_gen.class_indices
        
        value = {i for i in class_dic if class_dic[i]==predclass}
        
        return value
        
        



