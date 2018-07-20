**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 32 and 128 (model.py lines 64-68) 

The model includes RELU layers to introduce nonlinearity (code line 64-74), and the data is normalized in the model using a Keras lambda layer (code line 62). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 71,73,75). 

The model was trained and validated on different data sets to ensure that the model was not overfitting.
 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 78).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, 
driving the road in oposite directionnd  repeated hard coners two times.
For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to design a convolution neural network 

My first step was to use a convolution neural network model similar to the lenet5 I thought this model might be appropriate because its simple and can be build upon.

The architecture of lenet5 can be modified easily through adding some new feature that was indroduced with modern architectures. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that increase the depth of the model to reach 5 convolution layers and three fuly connected layers and add some dropout layers. 

This approach provided me with good enough model that can drive in simulation. In real life trial it will need more than just couple of thousand or simple model like mine beacause it will have a huge amount of corner cases to cover and overcome.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track so to improve the driving behavior in these cases, I increased the samples of corners and decreased the sample of straight line driving.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 61-74) consisted of a convolution neural network with the following layers and layer sizes

1- normalizing the data.
2- Cropping the region(70,25), (0,0)) to better focus on the road
3- Convolution layer with filter size = 24, kernel_size= ( 5, 5), strides = (2,2) and relu activation
4- Convolution layer with filter size = 36, kernel_size= ( 5, 5), strides = (2,2) and relu activation
5- Convolution layer with filter size = 48, kernel_size= ( 5, 5), strides = (2,2) and relu activation
6- Convolution layer with filter size = 64, kernel_size= ( 3, 3) and relu activation
7- Convolution layer with filter size = 64, kernel_size= ( 3, 3) and relu activation
8- Flatten the matrix into array
9- fully connected layer of length 100 and relu activation
10- dropout layer wiht 50% drop rate
11- fully connected layer of length 50 and relu activation
12- dropout layer wiht 50% drop rate
13- fully connected layer of length 10 and relu activation
14- dropout layer wiht 50% drop rate
15- fully connected layer of length 1 and relu activation

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior,

I first recorded two laps on track one using center lane driving.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to how to return to the center of the road after drifting to the sides.

Then I repeated this process on track one for one lap but in opposite direction to don't give a favor to certain direction.

To augment the data sat, I also flipped images and angles thinking that this would give me more train data and simulate for scenarios
that can't be generated in the simulator.

After the collection process, I had 30k of data points. I then preprocessed this data by normalizing and resize it to focus on the road.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as when the number of eposhs incease the validation error increase. I used an adam optimizer so that manually training the learning rate wasn't necessary.
