#!/usr/bin/env python3
# # Traffic Light Classifier
# ---
# 
# In this project, you’ll use your knowledge of computer vision
# techniques to build a classifier for images of traffic lights! You'll
# be given a dataset of traffic light images in which one of three lights
# is illuminated: red, yellow, or green.
# 
# In this notebook, you'll pre-process these images, extract features
# that will help us distinguish the different types of images, and use
# those features to classify the traffic light images into three classes:
# red, yellow, or green. The tasks will be broken down into a few sections:
# 
# 1. **Loading and visualizing the data**. 
#       The first step in any classification task is to be familiar
#       with your data; you'll need to load in the images of traffic
#       lights and visualize them!
# 
# 2. **Pre-processing**. 
#     The input images and output labels need to be standardized. This
#     way, you can analyze all the input images using the same
#     classification pipeline, and you know what output to expect when
#     you eventually classify a *new* image.
#     
# 3. **Feature extraction**. 
#     Next, you'll extract some features from each image that will help
#     distinguish and eventually classify these images.
#    
# 4. **Classification and visualizing error**. 
#     Finally, you'll write one function that uses your features to
#    classify *any* traffic light image. This function will take in an
#    image and output a label. You'll also be given code to determine
#    the accuracy of your classification model.
#     
# 5. **Evaluate your model**.
#     To pass this project, your classifier must be >90% accurate and
#     never classify any red lights as green; it's likely that you'll need
#     to improve the accuracy of your classifier by changing existing
#     features or adding new features. I'd also encourage you to try to
#     get as close to 100% accuracy as possible!
#     
# Here are some sample images from the dataset (from left to right: red,
# green, and yellow traffic lights):
# <img src="images/all_lights.png" width="50%" height="50%">

import cv2 # computer vision library
import helpers # helper functions
from collections import Counter
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # for loading in images
import unittest

# A class holding all tests
class Tests(unittest.TestCase):

    
    # Tests the `one_hot_encode` function, which is passed in as an argument
    def test_one_hot(self, one_hot_function):
        
        # Test that the generated one-hot labels match the expected one-hot label
        # For all three cases (red, yellow, green)
        try:
            self.assertEqual([1,0,0], one_hot_function('red'))
            self.assertEqual([0,1,0], one_hot_function('yellow'))
            self.assertEqual([0,0,1], one_hot_function('green'))
        
        # If the function does *not* pass all 3 tests above, it enters this exception
        except self.failureException as e:
            # Print out an error message
            print('*** TEST FAILED ***')
            print("Your function did not return the expected one-hot label.")
            print('\n'+str(e))
            return
        
        # Print out a "test passed" message
        print('*** TEST PASSED ***')
    
    
    # Tests if any misclassified images are red but mistakenly classified as green
    def test_red_as_green(self, misclassified_images):
        # Loop through each misclassified image and the labels
        for im, predicted_label, true_label in misclassified_images:
            
            # Check if the image is one of a red light
            if(true_label == [1,0,0]):
                
                try:
                    # Check that it is NOT labeled as a green light
                    self.assertNotEqual(predicted_label, [0, 0, 1])
                except self.failureException as e:
                    # Print out an error message
                    print('*** TEST FAIL ***')
                    print("Warning: A red light is classified as green.")
                    print('\n'+str(e))
                    return
        
        # No red lights are classified as green; test passed
        print('*** TEST PASSED ***')


# Image data directories
IMAGE_DIR_TRAINING = "traffic_light_images/training/"
IMAGE_DIR_TEST = "traffic_light_images/test/"

# Using the load_dataset function in helpers.py
# Load training data
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)


## TODO: Write code to display an image in IMAGE_LIST (try finding a yellow traffic light!)
## TODO: Print out 1. The shape of the image and 2. The image's label

# 2. Pre-process the Data
#
# After loading in each image, you have to standardize the input and
# output!
#
# ## Input
#
# This means that every input image should be in the same format, of
# the same size, and so on. We'll be creating features by performing the
# same analysis on every picture, and for a classification task like this,
# it's important that similar images create similar features!
#
# ## Output
#
# We also need the output to be a label that is easy to read and easy
# to compare with other labels. It is good practice to convert categorical
# data like "red" and "green" to numerical data.
# A very common classification output is a 1D list that is the length of
# the number of classes - three in the case of red, yellow, and green lights
# - with the values 0 or 1 indicating which class a certain image is. For
# example, since we have three classes (red, yellow, and green), we can
# make a list with the order: [red value, yellow value, green value]. In
# general, order does not matter, we choose the order [red value, yellow
# value, green value] in this case to reflect the position of each light
# in descending vertical order.
#
# A red light should have the label: [1, 0, 0]. Yellow should be:
# [0, 1, 0]. Green should be: [0, 0, 1]. These labels are called one-hot
# encoded labels.
#
# (Note: one-hot encoding will be especially important when you work
# with machine learning algorithms).


# Requirements
# +  Resize each image to the desired input size: 32x32px.
# +  (Optional) You may choose to crop, shift, or rotate the images in this step as well.
# It's very common to have square input sizes that can be rotated (and
# remain the same size), and analyzed in smaller, square patches. It's
# also important to make all your images the same size so that they can
# be sent through the same pipeline of classification steps!

def equalize(image):
    dimension=2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image[:,:,dimension] = cv2.equalizeHist(image[:,:,dimension])
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

# This function should take in an RGB image and return a new, standardized version
def standardize_input(image):
    
    ## TODO: Resize image and pre-process so that all "standard" images are the same size  
    standard_im = np.copy(image)
    standard_im = cv2.resize(standard_im, (32,32))
    
    return standard_im

## TODO: One hot encode an image label
## Given a label - "red", "green", or "yellow" - return a one-hot encoded label

# Examples: 
# one_hot_encode("red") should return: [1, 0, 0]
# one_hot_encode("yellow") should return: [0, 1, 0]
# one_hot_encode("green") should return: [0, 0, 1]

LABEL_ENCODING = {'red': [1,0,0],
                  'yellow': [0,1,0],
                  'green': [0,0,1]}

TUPLE_ENCODED = {(1,0,0): 'red',
                 (0,1,0): 'yellow',
                 (0,0,1): 'green'}

def one_hot_encode(label):    
    return LABEL_ENCODING[label]

def label_from_encoding(encoding):
    return TUPLE_ENCODED[tuple(encoding)]

# Importing the tests
tests = Tests()

# Test for one_hot_encode function
tests.test_one_hot(one_hot_encode)

def standardize(image_list):
    
    # Empty image data array
    standard_list = []

    # Iterate through all the image-label pairs
    for item in image_list:
        image = item[0]
        label = item[1]

        # Standardize the image
        standardized_im = standardize_input(image)

        # One-hot encode the label
        one_hot_label = one_hot_encode(label)    

        # Append the image, and it's one hot encoded label to the full, processed list of image data 
        standard_list.append((standardized_im, one_hot_label))
        
    return standard_list

# Standardize all training images
STANDARDIZED_LIST = standardize(IMAGE_LIST)

RED_IMAGES = STANDARDIZED_LIST[:723]
YELLOW_IMAGES = STANDARDIZED_LIST[723:758]
GREEN_IMAGES = STANDARDIZED_LIST[758:]

# pick a random image from each class to test on
SAMPLE_RED = random.choice(RED_IMAGES)
SAMPLE_GREEN = random.choice(GREEN_IMAGES)
SAMPLE_YELLOW = random.choice(YELLOW_IMAGES)

## TODO: Create a brightness feature that takes in an RGB image and outputs a feature vector and/or value
## This feature should use HSV colorspace values
def average_brightness(img):
    return np.sum(img) / (img.shape[0] * img.shape[1])

# This feature relies on band matching, where 0=red, 1=yellow, 2=green.
SCORE_LABELS = ['red', 'yellow', 'green']

def max_band(bands):
    max_i = 0
    max_v = 0
    for i in range(len(bands)):
        if bands[i] > max_v:
            max_v = bands[i]
            max_i = i
    return max_i

def min_band(bands):
    min_i = 0
    min_v = float('+inf')
    for i in range(len(bands)):
        if bands[i] < min_v:
            min_v = bands[i]
            min_i = i
    return min_i


def image_bands(image, dimension):
    band1 = image[5:15,7:22,dimension]
    band2 = image[10:20,7:22,dimension]
    band3 = image[20:30,7:22,dimension]
    return band1, band2, band3
        
def create_feature(rgb_image, plot=False):
    
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    dimension = 2 # V
    band1, band2, band3 = image_bands(hsv_image, dimension)
    
    prediction = max_band(list(map(average_brightness, [band1, band2, band3])))
    return SCORE_LABELS[prediction]

print('Average brightness of sample red: ', create_feature(SAMPLE_RED[0]))
print('Average brightness of sample yellow: ', create_feature(SAMPLE_YELLOW[0]))
print('Average brightness of sample green: ', create_feature(SAMPLE_GREEN[0]))

def color_band(rgb_image):
    image = rgb_image[5:27, 7:22, :]    
    image = equalize(image)
    band_brightness = lambda bands : list(map(average_brightness, bands))
    bands_red = min_band(band_brightness(image_bands(image, 0)))
    bands_green = min_band(band_brightness(image_bands(image, 1)))
    bands_blue = min_band(band_brightness(image_bands(image, 2)))
    
    band = [bands_red, bands_green, bands_blue]
    print(band)
    return min_band(band)

print('Color prediction for sample red:', color_band(SAMPLE_RED[0]))
print('Color prediction for sample yellow:', color_band(SAMPLE_GREEN[0]))
print('Color prediction for sample green:', color_band(SAMPLE_YELLOW[0]))

# This function should take in RGB image input
# Analyze that image using your feature creation code and output a one-hot encoded label
def estimate_label(rgb_image):
    scores = Counter()
    max_band_brightness = create_feature(rgb_image)
    scores[max_band_brightness] += 2
    
    best_score = scores.most_common(1)[0]
    return one_hot_encode(best_score[0])

# Using the load_dataset function in helpers.py
# Load test data
TEST_IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TEST)

# Standardize the test data
STANDARDIZED_TEST_LIST = standardize(TEST_IMAGE_LIST)

# Shuffle the standardized test data
random.shuffle(STANDARDIZED_TEST_LIST)

# Constructs a list of misclassified images given a list of test images and their labels
# This will throw an AssertionError if labels are not standardized (one-hot encoded)

def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]
        assert(len(true_label) == 3), "The true_label is not the expected length (3)."

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)
        assert(len(predicted_label) == 3), "The predicted_label is not the expected length (3)."

        # Compare true and predicted labels 
        if(predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))
            
    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


# Find all misclassified images in a given test set
MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

# Accuracy calculations
total = len(STANDARDIZED_TEST_LIST)
num_correct = total - len(MISCLASSIFIED)
accuracy = num_correct/total

print('Accuracy: ' + str(accuracy))
print("Number of misclassified images = " + str(len(MISCLASSIFIED)) +' out of '+ str(total))

if (len(MISCLASSIFIED) > 0):
    # Test code for one_hot_encode function
    tests.test_red_as_green(MISCLASSIFIED)
else:
    print("MISCLASSIFIED may not have been populated with images.")
