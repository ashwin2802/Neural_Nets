#import keras libraries
from keras.models import model_from_json
from keras.utils import to_categorical
#import MNIST database
from keras.datasets import mnist

#download the MNIST data
(X,Y),(x_t,y_t) = mnist.load_data()

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
Y = to_categorical(Y)
y_t = to_categorical(y_t)

# evaluate the model
def select():
    bin = int(input("Enter 0 to evaluate model on training set and 1 to evaluate on the test set: "))
    if(bin==1):
        scores = model.evaluate(x_t, y_t, verbose=0)
        print("Printing accuracy of prediction for each image. Press ^C to stop.")
        print("{0:s}: {1:.2f}".format(model.metrics_names[1], scores[1]*100))
    elif(bin==0):
        scores = model.evaluate(X, Y, verbose=0)
        print("Printing accuracy of prediction for each image. Press ^C to stop.")
        print("{0:s}: {1:.2f}".format(model.metrics_names[1], scores[1]*100))
    else:
        print("Try again.")
        select()

select()
