import numpy as np
#import keras libraries
from keras.models import model_from_json
from keras.utils import to_categorical
#import MNIST database
from keras.datasets import mnist

#download the MNIST data
(X,Y),(x_t,y_t) = mnist.load_data()
print("Loaded database.")

# load json and create model
json_file = open('./model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("./model/model.h5")
print("Loaded model from disk.")

#reshape data to fit into model
X = X.reshape(60000,28,28,1)
x_t = x_t.reshape(10000,28,28,1)

#one-hot encode the labels
y = to_categorical(Y)
Y_t = to_categorical(y_t)

#extract predicted label from model.predict
def extract(arp):
    arr = list(np.array(list(arp[0].flat)))
    for i in range(len(arr)):
        if(arr[i]==max(arr)):
            return i

#output predicted label and compare with database label
def output(a, b):
    print("The digit in the image is " + str(a))
    print("The label of the image is " + str(b))
    if(a==b):
        print("Prediction successful.")
    else:
        print("Prediction failed.")

#prediction function with error handling
def predict():
    num = int(input("Enter the ID of image to test: "))
    if(num<10001 and num>0):
        out = model.predict(x_t[num-1][None,:,:,:])
        output(extract(out), y_t[num-1])
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


