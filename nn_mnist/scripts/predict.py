import numpy as np
#import keras libraries
from keras.models import model_from_json
from keras.utils import to_categorical
#import MNIST database
from keras.datasets import mnist

#function to allow single image input in the predict function
def prep(x):
    x = np.expand_dims(x,axis=0)

#download the MNIST data
(X,Y),(x_t,y_t) = mnist.load_data()
print("Loaded database.")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk.")

#reshape data to fit into model
X = X.reshape(60000,28,28,1)
x_t = x_t.reshape(10000,28,28,1)

#one-hot encode the labels
y = to_categorical(Y)
Y_t = to_categorical(y_t)

#prediction function with error handling
## incomplete!
def predict():
    num = int(input("Enter the ID of image to test: "))
    if(num<10001 and num>0):
        model.predict(prep(x_t[num]))
    else:
        print("Out of range. Please enter a number between 1 and 10000.")
        predict()

#execute with error handling
def run():
    j = int(input("Enter number of images to test: "))
    if(j>0 and j<10001):
        for i in range(j):
            predict()
    else:
        print("Please enter a number between 1 and 10000.")
        run()

run()

