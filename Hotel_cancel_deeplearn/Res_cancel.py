# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 17:32:59 2024

@author: Mariana Khachatryan
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

from tensorflow import keras
from tensorflow.keras import layers


df = pd.read_csv('Hotel_Reservations.csv')


df.info()



"""
Dataset Description
The file contains the different attributes of customers' reservation details. The detailed data dictionary is given below
Booking_ID: unique identifier of each booking
No of adults: Number of adults
No of children: Number of Children
noofweekend_nights: Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel
noofweek_nights: Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel
typeofmeal_plan: Type of meal plan booked by the customer:
requiredcarparking_space: Does the customer require a car parking space? (0 - No, 1- Yes)
roomtypereserved: Type of room reserved by the customer. The values are ciphered (encoded) by INN Hotels.
lead_time: Number of days between the date of booking and the arrival date
arrival_year: Year of arrival date
arrival_month: Month of arrival date
arrival_date: Date of the month
Market segment type: Market segment designation.
repeated_guest: Is the customer a repeated guest? (0 - No, 1- Yes)
noofprevious_cancellations: Number of previous bookings that were canceled by the customer prior to the current booking
noofpreviousbookingsnot_canceled: Number of previous bookings not canceled by the customer prior to the current booking
avgpriceper_room: Average price per day of the reservation; prices of the rooms are dynamic. (in euros)
noofspecial_requests: Total number of special requests made by the customer (e.g. high floor, view from the room, etc)
booking_status: Flag indicating if the booking was canceled or not.

"""
#%%

df_hotel=df.copy()
#Assigning numeric labels to classes
df_hotel["booking_status"]=df_hotel["booking_status"].map({"Not_Canceled":0,"Canceled":1})

#Remove "booking status" from df_hotel and save series in y
y=df_hotel.pop("booking_status")


#We can also drop the column Booking_ID in further analysis
X = df_hotel.drop("Booking_ID", axis=1)


#We will use One_Hot Encoding for categorical variables
info_categ=["market_segment_type","room_type_reserved","type_of_meal_plan"]

#And we'll scale numerical info better performance of the model
info_numeric=['no_of_adults', 'no_of_children', 'no_of_weekend_nights',
       'no_of_week_nights', 'required_car_parking_space', 'lead_time', 
       'arrival_year', 'arrival_month','arrival_date', 'repeated_guest',
       'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
       'avg_price_per_room', 'no_of_special_requests']


transformer_categ = make_pipeline( OneHotEncoder(handle_unknown='ignore'))
transformer_numeric = make_pipeline( StandardScaler())
preprocessor = make_column_transformer( (transformer_categ, info_categ),
                                       (transformer_numeric, info_numeric))


# stratify=y - if variable y is a binary categorical variable with 
#values 0 and 1 and there are 25% of zeros and 75% of ones, stratify=y will
# make sure that your random split has 25% of 0's and 75% of 1's.
X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, train_size=0.75)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)

inp_shape=[X_train.shape[1]]

#%%

#Defining deep learning model with binary classification

model = keras.Sequential([
    layers.BatchNormalization(input_shape=inp_shape), #to standardize raw input variables: transform inputs so that they will have a mean of zero and a standard deviation of one.
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(), #to standardize the outputs of a hidden layer.
    layers.Dropout(0.3), # this applies 30% dropout to the next layer
    layers.Dense(256, activation='relu'),    
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid'),
])


#%%
#Compiling the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)


#%%

# train the model and study results

#Use early stopping to stop when validation loss stops increasing
early_stopping = keras.callbacks.EarlyStopping(
    patience=7, #number of epochs before stopping if no improvement
    min_delta=0.001, # amount of change counted as improvement
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=500,
    epochs=300,
    callbacks=[early_stopping],
)

#%%

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy",ylim=(0.25,0.3))
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy",ylim=(0.87,0.89))

""" We can see that the model performed well and stopped trining when vlidation
cross entropy stopped decreasing, and Keras stopped well before 300 epochs
Also looking at the accuracy plot we can see that it was increasing when cross
entropy was decreasing and keras stopped when accuracy started to decrease for validation
"""
