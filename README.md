# **Behavioral Cloning**

## Kavan Brandon

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn-architecture-624x890.png "Model Visualization"
[image2]: ./examples/center_2018_05_12_12_05_34_829.jpg "Center Lane Driving"
[image3]: ./examples/left_2018_05_12_12_07_37_476.jpg "Left Camera Image"
[image4]: ./examples/right_2018_05_12_12_06_45_758.jpg "Right Camera Image"
[image5]: ./examples/left_2018_05_12_12_07_37_476_flipped.jpg "Flipped Image"

---
### Required Files

#### 1. Project files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* video.mp4 for showing the recorded drive in autonomous mode

### Quality of Code

#### 1. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 2. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the Nvidia Deep Learning for Self-Driving Cars network architecture. The architecture consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers.

#### 2. Attempts to reduce overfitting in the model

The training data was split into test and validation sets (80% and 20% respectively). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. The script also uses a generator for preprocessing the data sets.

#### 3. Model parameter tuning

The model uses an adam optimizer.

#### 4. Appropriate training data

The training data was recorded by making two laps around the track. The first lap primarily focused on center lane driving while the second lap introduced more driving variance (ex. driving on the right side of the lane then moving back towards the center).

### Architecture and Training Documentation

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was providing adequate collection data that includes different types of driving activity. I tried to mimic different driving patterns such as staying in the center of the lane, moving away from the center then back to the center, and using wide and sharp turn angles. The reasoning for this approach was to allow the model architecture to learn from as many scenarios as possible.

After reading about Nvidia's Deep Learning for Self-Driving Cars network architecture, I decided to use that structure on my collection data for training purposes. The architecture's use of CNNs means the model can learn driving angles based on the image recognition in collection data. It seemed like an appropriate model that has been tested previously in the field. Furthermore, the model is designed to be powerful enough to follow lane and road boundaries without the need for computer vision marking detection or path planning algorithms.

I used a generator to pre-process the images before training commenced. The generator shuffles the csv data before preprocessing begins. The first step was improving the quantity of training data by adding left and right images to the datasets in addition to the center image. This allows the model to understand the angle differences when viewing the road from the left and right cameras. Additionally, I added additional steering measurements to the sets by adding a correction of 0.25 to the left image and -0.25 to the right image. Lastly, the generator converts the images to from BGR to RGB.

Data augmentation is introduced in the script as well. Using the CV2 library, I augmented the images by flipping them in addition to their corresponding measurements. As described in the lesson, flipping the images is useful for removing turn bias within the model.

In order to gauge how well the model was working, the data is split into training and validation sets (80% and 20% respectively). The model seemed to have a low mean squared error on the training set and validation set which seemed to imply the model wasn't overfitting.

Running the model through the simulator, the car typically stayed in the center of the lane. On sharper turns, the car did veer closer to the right and left lane markings, but would move back into the center of the lane after the road straightened. At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

![alt text][image1]

As seen in the diagram above, I used the Nvidia Deep Learning for Self-Driving Cars network architecture.

* The architecture consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers.

* The first 3 convolutional layers include 5x5 kernel sizes while the last two include 3x3 kernel sizes.

* The convolutional layers include RELU activations to introduce nonlinearity.

* The data is normalized in the model using a Keras lambda layer described in the lesson. Furthermore, the model introduces cropping to remove unnecessary image data during the training process. As described in the lesson, it isn't necessary to have the hood of the vehicle or trees in the horizon included in the images, thus it is necessary to crop out those areas. The model introduces three 0.5 dropout layers (model.py lines 81, 83, and 85) to address the potential for overfitting.

* The model uses an adam optimizer. The adam optimizer is an extension to stochastic gradient descent and didn't require further tuning while using Keras.
* Uses 5 epochs for training and a batch size of 64

#### 3. Creation of the Training Set & Training Process

To capture driving data, I made two laps around the track. The first lap primarily consisted of center lane driving throughout all portions of the track. This is an example of center lane driving:

![alt text][image2]

The second lap around the track consisted of recovery driving. Especially around sharper turns, I would take wider movements and then bring the car back to the center. Below is a left camera and right camera image around the sharper turns.

![alt text][image3]
![alt text][image4]

In terms of data augmentation, images were flipped to combat against left and right turn bias. Here is an example of a flipped image:

![alt text][image5]

While using the fit_generator method described in the lesson, I multiplied the number of training and validation samples by 5 to give the model a larger amount of data points to train. It seemed that providing more data points enhanced the performance of the model considerably over each epoch.

### Simulation

#### 1. Test Data Navigation

The file video.mp4 shows the results of running the model.h5 in autonomous simulation. The car is able to successfully navigate track 1 although the car has difficulty staying perfectly centered in the lane during sharper turns.
