#import keras libraries
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.models import model_from_json

#import the MNIST database
from keras.datasets import mnist

#download MNIST data and split it into training and testing sets
(X_train,y_train), (X_test, y_test) = mnist.load_data()

#reshape data to fit the model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000, 28, 28,1)

#one-hot encode target columns
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#create the model
model = Sequential()

#add layers to the model
model.add(Conv2D(64, kernel_size = 3, activation = 'relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size = 3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#compile the model using accuracy to measure performance
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk.")
