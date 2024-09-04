## __Facemask image classification__

In this project I have used convolutional neural networks from keras.tensorflow to classify weather the person in the image wears a facemask.
I first create training and testing folders each containg two folders of people images with and without a facemask.
I use Sequential model from tensorflow.keras with 3 convolutional modules followed by flattening layer to remove all of the dimensions except for one,
then a dense layer of 128 neurons, followed by a random dropout of 50% of outputs and followed by a Dense layer with singe neuron 
since we expect only one output for classification. 
We then compile model using "binary_crossentropy" for loss.
I then resize images to an average height and use data generators to read the pictures and store in float32 tensors.
We then fit the model to images and define early stopping based on validation loss.
WIth this model I have achieved 94% accuracy.
