# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[dataVisualization]: ./output_images/dataVisualization.jpg "dataVisualization"

[dataDistribution]:     ./output_images/dataDistribution.jpg "dataDistribution"
[augmentedDataDistribution]:     ./output_images/augmentedDataDistribution.jpg "augmentedDataDistribution"

[originalAndGeneratedImages]:     ./output_images/originalAndGeneratedImages.jpg "originalAndGeneratedImages"
[originalAndPreprocessedImages]:     ./output_images/originalAndPreprocessedImages.jpg "originalAndPreprocessedImages"

[onlineTrafficSignImages]: ./output_images/onlineTrafficSignImages.jpg "Online Traffic Sign Immges"
[onlineTrafficSignImagesPreprocessed]: ./output_images/onlineTrafficSignImagesPreprocessed.jpg "onlineTrafficSignImagesPreprocessed"
[onlineTrafficSignImagesPredictions]: ./output_images/onlineTrafficSignImagesPredictions.jpg "onlineTrafficSignImagesPredictions"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/kanerodriguez/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

* The size of training set is 34799
* The size of the validation set is ?
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 with 3 channels
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of the types of classifiers found in the training, validation, and testing image sets.

![dataVisualization][dataVisualization]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

First, I decided to convert the images to grayscale because I wanted to ignore the noise that can come with color. In this case, the majority of the image information to be gained lies within the shape of the traffic sign, the shape of the images depicted on the sign, and any text on the sign. I also normalized the image, per the projects recommendations, because doing so lead to a mean closer to zero for the training data. In practice, this helps speed up the time to train our model. 

<b>Note</b>: color information can be useful, but processing 3 channels provides an increasingly computationally complex problem than a 1 channel image!

An example of a traffic sign image before and after preprocessing is given below:

|        Original Image           |    Preprocessed Image                   |
|:-------------------------------:|:---------------------------------------:|
| ![originalImage][originalImage] | ![preprocessedImage][preprocessedImage] |

To minimize the variance of the data, additional data was generated. 

To do this, I decided to generate new images from currently existing images for each classifier in the training set that falls below the median classifier bin size.

I measured variance based on the frequency of each label in the training set; I calculated the median amount of images per label, and I chose to have each current image contribute a fraction of the difference between it's repective bin size and the median. The resulting distribution of this process produces the chart below.

The steps to generating new images involved two modifications. First, I randomly rotate the image anywhere from -15 to 15 degrees. Then, I warp the image to create a "zoomed in" image. I did this because in my initial tests I found that my model did not detect online images that were at an angle very well, and I hoped that augmenting the training data set with traffic signs in different orientations would make the model robust to these instances.

Here is an example of an original image and a generated image:

|        Original Image           |    Augmented Image                      |
|:-------------------------------:|:---------------------------------------:|
| ![originalImage][originalImage] | ![generatedImage][generatedImage]       |

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

My final model used an Adam Optimizer, a 128 image batch size, 15 epochs, and a 0.0001 learning rate.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

##### Initial Solution (LeNet-5)

My initial solution to the problem involved a LeNet-5 architecture applied to normalized (0.0 to 1.0) gray scaled (32x32x1) images. This resulted in an accuracy of 89% on the validation set. To increase accuracy, I initially chose to randomly rotate 25% of the training images by a multiple of 90 degrees. Adding random rotations resulted in an 82% accuracy on the validation set. This could be beneficial if the input images are expected to contain traffic signs at a large range of angles, but data visualizations of the testing and validation sets show that this is not the case. Also, traffic signs that depict turning in one direction and are now pointing to the opposite direction could be one cause of the reduction in accuracy.

##### Solution 2 (SimpNet)

I decided to rewrite my model to follow the SimpNet architecture described in [this](https://arxiv.org/pdf/1802.06205.pdf) paper. The paper presents a deep convolusional network that uses less parameters and operations while still empirically achieving state-of-the-art on standard data sets (i.e. MNST). They discussed the benefits of max-pooling over strided convolutions, they proposed "SAF-pooling" (essentially max-pooling followed by dropout), and why not to use 1x1 convolutions or fully connected layers in the beginning of an architecture. Unfortunately, my implementation resulted in a model with so many layers and parameters that I was unable to run the model on my puny laptop...

##### Solution 3 (Augmenting the Data Set)

Moving forward, I reverted back to the LeNet architecture and prioritized the data itself. I changed the way I was normalizing my data so that it would have a mean closer to zero (1.0 to 0.32). Then, I generated new data from preexisting data based on which classifier bins were below the mean bin size. This resulted in my lowering the variance of my data by 56% and improving my validation set accuracy to 92.8%.

##### Solution

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

##### Activation Function

Rectified Linear Unit's (ReLU's) are used for the activation functions because it is less computationally complex than other activation functions, such as the scaled hyperbolic tangent function.

##### Future Considerations

A future modification to the preprocessing step involves utilizing the 'coords' data that specifies where a traffic sign is found within an image. With these coordinates, one can decide to initialize the weights to reflect this priotization instead of starting the weights at random or perhaps one can autogenerate more images from the road signs found in these images. I attempted to utilize the coordinates in generating new data, and I was unable to make use of them since they were not also scaled with the data that they came with.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![onlineTrafficSignImages][onlineTrafficSignImages]

Here is 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|

The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This did worse than the 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

![onlineTrafficSignImagesPredictions][onlineTrafficSignImagesPredictions]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?