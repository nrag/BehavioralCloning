Behavrioal Cloning Project

The goals / steps of this project are the following:
Use the simulator to collect data of good driving behavior
Build, a convolution neural network in Keras that predicts steering angles from images
Train and validate the model with a training and validation set
Test that the model successfully drives around track one without leaving the road
Summarize the results with a written report

[//]: # (Image References)

[image1]: ./images/vgg16.png "Model Visualization"
[image2]: ./images/driving.jpg "Grayscaling"
[image3]: ./images/recovery1.jpg "Recovery Image"
[image4]: ./images/recovery2.jpg "Recovery Image"
[image5]: ./images/recovery3.jpg "Recovery Image"
[image6]: ./images/driving.jpg "Normal Image"
[image7]: ./images/driving_flipped.jpg "Flipped Image"
[image8]: ./images/throttling_distribution.png "Distribution of throttling"
[image9]: ./images/throttling_distribution_balance.png "Distribution of throttling"
[image10]: ./images/model.png "Distribution of throttling"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality
####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* vgg16.py containing the script to create the model
* train_car_driving.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* candidate.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py candidate.h5
```

####3. Submssion code is usable and readable

The train_car_driving.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with VGG-16 as the basis (vgg16.py). I introduced Dropout layers in between the Dense layers and also at the end of the last two blocks of the convolutional layers.


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (vgg16.py). The loss function also contains L2 regularization on the weights to avoid overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 26-30). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of 

Center lane driving
Recovering from the left and right sides of the road
driving on the curves
Mirror images of the images above

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the VGG-16 model. I thought this model might be appropriate because VGG-16 is quite flexible and is very successful with the ImageNet data. I thought the features extracted would be very useful for the behavioral cloning example.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a comparable loss on training and validation sets. But the model didn’t perform very well around the curves. I realized that this is because the dataset has a lot more images from straight-line driving than curves.

To alleviate this problem, I collected data only while driving around the curves. This helped the model perform better. But I realized that the model was not immune to small perturbations. To fix this, I collected data to recover from left and right sides of the roads. I also added mirror images of the images screenshots.

To combat the overfitting, I added dropout layers, regularization and created a large dataset.


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road for over an hour.

####2. Final Model Architecture

I started with the VGG-16 architecture shown below
Here is a visualization of the VGG-16 architecture:
![alt text][image1]

I made the following modifications to the VGG-16 architecture:
* Replaced the last 1000-node fully-connected layer with softmax activation with a 1-node fully connected layer with linear activation
* Added Dropouts after fully connected layer
* Added Dropouts after the last two convolutional blocks (because GTX1080 ran out of memory with dropouts after other convolutional blocks. I could have reduced my batch size but the training time would have shot up)

The final model architecture  consisted of a convolution neural network with the following layers and layer sizes. Here's the visualization of the model

![alt text][image10]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded about 40,000 on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to center the car. These images show what a recovery looks like starting from near the lane edges.

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data set, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

To ensure that the data set is not skewed towards straight line driving, I collected about 10,000 samples while driving around the curves.

After the collection process, I had about 67,000 number of data points. I then preprocessed this data by normalizing the data.
Here's the distribution of samples after I collected all the data.
![alt text][image8]

I should have randomly sampled the images with throttling as zero. This would have made the sample distribution similar to this:
![alt text][image9]

I finally randomly shuffled the data set and put 33% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20 as evidenced by the tapering loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
