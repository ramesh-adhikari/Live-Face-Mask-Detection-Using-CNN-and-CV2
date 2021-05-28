# Live Face Mask Detection Using CNN and CV2

- The corona virus Outbreak has created various changes in the lifestyle of everyone around the world. 
- In those changes wearing a mask has been very vital to every individual and same has been announced by government and WHO. 
- Detection of people who are not wearing masks is a challenge due to the large number of populations. 
- This project can be used in schools, hospitals, banks, airports etc as a digitalized scanning tool. 

## Dataset
- The dataset are separated in train and test directory.
- Train directory contains images of with_mask:658 image and without_mask:657
- Test directory contains images of with_mask:97 image and without_mask:97

## Library Used In this project as
- numpy: Is a Python library used for working with arrays.
- keras: Is a deep learning API written in Python, running on top of the machine learning platform TensorFlow. 
- sklearn:  Is a library contains a lot of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction.
- cv2:  uses machine learning algorithms to search for faces within a picture. Because faces are so complicated, there isn’t one simple test that will tell you if it found a face or not. Instead, there are thousands of small patterns and features that must be matched. The algorithms break the task of identifying the face into thousands of smaller, bite-sized tasks, each of which is easy to solve. These tasks are also called classifiers.
- matplotlib: Is a cross-platform, data visualization and graphical plotting library for Python and its numerical extension NumPy.
- Convolutional Neural Network is used to train the model.




## To run this project
1. Clone or download the zip file
2. go to face-mask directory
3. Then run train the cnn model script by: python3 train_cnn_model.py
![image](https://user-images.githubusercontent.com/training_model_accuracy_and_loss.png)
fig: training CNN model
3. After completion of training the cnn model run: python3 test_with_live_video.py

