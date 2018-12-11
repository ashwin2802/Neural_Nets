#import matplotlib
import matplotlib.pyplot as plt
#import MNIST database
from keras.datasets import mnist

#download the MNIST data
(X,Y), (x,y) = mnist.load_data()

#show the expected image and labels from the training set and its shape
def chk():
    num = int(input("Enter the ID of image to show: "))
    if(num>=1 and num<60001):
        print("Using training database.")
        print("Image number " + str(num) + " is:")
        plt.imshow(X[num-1])
        print("Shape of the image is: ")
        X[num-1].shape
        print("Image shows the number: " + str(Y[num-1]))
    elif(num>=60001 and num<70000):
        print("Using test database.")
        print("Image number " + str(num) + " is:")
        num = num - 60001
        plt.imshow(x[num-1])
        print("Shape of the image is: ")
        x[num-1].shape
        print("Image shows the number: " + str(y[num-1]))
    else:
        print("Please enter a number between 1 and 70000.")
        chk()

def run():
    j = int(input("Enter the number of images to show: "))
    if(j>0 and j<70001):
        for i in range(j):
            chk()
    else:
        print("Please enter a number between 1 and 70000.")
        run()

run()
