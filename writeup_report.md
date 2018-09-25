# **Traffic Sign Recognition** 

## Writeup

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

[onlineTrafficSignImagesBeforeAndAfterPreprocessing]: ./output_images/onlineTrafficSignImagesBeforeAndAfterPreprocessing.jpg "onlineTrafficSignImagesBeforeAndAfterPreprocessing"

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
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 with 3 channels
* The number of unique classes/labels in the data set is 43

Each type of traffic sign in the data set is shown below:

![dataVisualization][dataVisualization]

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of the types of classifiers found in the training, validation, and testing image sets.

![dataDistribution][dataDistribution]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

First, I decided to convert the images to grayscale because I wanted to ignore the noise that can come with color. In this case, the majority of the image information to be gained lies within the shape of the traffic sign, the shape of the images depicted on the sign, and any text on the sign. I also normalized the image, per the projects recommendations, because doing so leads to a mean closer to zero for the training data. In practice, this helps speed up the time to train our model. 

<b>Note</b>: Color information can be useful, but processing 3 channels provides an increasingly computationally complex problem than a 1 channel image!

An example of a traffic sign image before and after preprocessing is given below:

![originalAndPreprocessedImages][originalAndPreprocessedImages]

To minimize the variance of the data, additional data was generated. 

I generated new images from preexisting images for each classifier in the training set. New images were generated for each labeled traffic sign until all traffic signs had 150% the number of images as the previous traffic sign label with the highest frequency. This resulted in a training data set boost from 34,799 images to 129,797 images.

The resulting classifier distribution in the augmented data set is given in the chart below.

![augmentedDataDistribution][augmentedDataDistribution]

The steps to generating new images involved three modifications. First, I convert the image into HLS format and alter the brightness of each pixel by anywhere from 80% to 120% of the original pixel. I convert the image back into RGB format, and I randomly rotate the image anywhere from -10 to 10 degrees. Finally, I randomly translate the image in the x or y direction anywhere from -2 to 2 pixels.

Here is an example of an original image and a generated image:

![originalAndGeneratedImages][originalAndGeneratedImages]

At one point in time I tried to apply a shear and scale, and I found that doing so only decreased my accuracy. This could have been a result of how much I was shearing or how much larger or smaller I was making the image. I stuck with translating, rotating, and brightening because this provided enough variance in the generated images to make each somewhat unique while not conributing too much noise to the data set. (i.e. these modifications did not cause the traffic signs to become unrecognizable)

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

| Layer # |  Layer Type | Filter | Stride | Input | Output |
|:-------:|:-----------:|:------:|:------:|:-----:|:------:|
| 1 |  Convolusion | 5 x 5 x 6 | 1 x 1 | 32 x 32 x 1 | 28 x 28 x 6 |
| 2 |  Max Pool | 2 x 2 | 2 x 2 | 28 x 28 x 6 | 14 x 14 x 6 |
| 3 |  Convolusion | 5 x 5 x 16 | 1 x 1 | 14 x 14 x 6 | 10 x 10 x 16 |
| 4 |  Max Pool | 2 x 2 | 2 x 2 | 10 x 10 x 16 | 5 x 5 x 16 |
| 5 |  Convolusion | 5 x 5 x 400 | 1 x 1 | 5 x 5 x 16 | 1 x 1 x 400 |
| 6 | Flatten Combine | N/A | N/A | Layer 4 output of (5 x 5 x 16) and Layer 5 output of (1 x 1 x 400) | 800 |
| 7 |  Fully Connected Layer | N/A | N/A | 800 | 43 |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

TODO: My final model used an Adam Optimizer, a 128 image batch size, 20 epochs, and a 0.0001 learning rate.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

##### Initial Solution (LeNet-5)

My initial solution to the problem involved a LeNet-5 architecture applied to normalized (0.0 to 1.0 pixel values) gray scaled (32x32x1) images. This resulted in an accuracy of 89% on the validation set. To increase accuracy, I initially chose to randomly rotate 25% of the training images by a multiple of 90 degrees. Adding random rotations resulted in an 82% accuracy on the validation set. This could be beneficial if the input images are expected to contain traffic signs at a large range of angles, but data visualizations of the testing and validation sets show that this is not the case. Also, traffic signs that depict turning in one direction and are rotated 180 degrees in the opposite direction could be one cause of the reduction in accuracy.

##### Solution 2 (SimpNet)

I decided to rewrite my model to follow the SimpNet architecture described in [this](https://arxiv.org/pdf/1802.06205.pdf) paper. The paper presents a deep convolusional network that uses less parameters and operations while still empirically achieving state-of-the-art on standard data sets (i.e. MNST). They discussed the benefits of max-pooling over strided convolutions, they proposed "SAF-pooling" (essentially max-pooling followed by dropout), and why not to use 1x1 convolutions or fully connected layers in the beginning of an architecture. Unfortunately, this 'minimal' solution resulted in a model with so many layers and parameters that I was unable to run the model on my puny laptop. This could be a result of my incorrect implementation, and I may possibly revisit this in the future.

##### Solution 3 (Augmenting the Data Set)

Moving forward, I reverted back to the LeNet architecture and decided to prioritize the data itself. I changed the way I was normalizing my data so that it would have a mean closer to zero (1.0 to 0.32). Then, I generated new data from preexisting data based on which classifier bins were below the mean bin size. This resulted in my lowering the variance of my data by 56% and improving my validation set accuracy to 92.8%.

##### Solution 4 (Multi-Scale Convolusional Neural Network)

In revisiting the architecture component of the solution, I decided to create a new architecture adapted from the one described in [this](https://www.researchgate.net/publication/224260345_Traffic_sign_recognition_with_multi-scale_Convolutional_Networks) paper and previously used by [this](https://github.com/jeremy-shannon/CarND-Traffic-Sign-Classifier-Project) student. The major change from this method and the LeNet method was that the outputs from my second convolusional / maxpool layer were fed into two different components. The first output fed into a convolusional filter and this output was flattened and combined with the flattened outputs of the previous layer. Furthermore, this output was connected to only one fully connected layer. This differs from LeNet, which ended in multiple fully connected layers. 

This change resulted in a 93.3% accuracy on the validation set!

##### Final Solution (Massive data generation)

To achieve an even higher accuracy, I lowered the variance of the data set from 383681.36 to 4.69 by bringing the number of training images up from 34,799 to 129,797. I generated images by randomly rotating, translating, and brightening preexisting images. This resulted in a validation set accuracy of 94.9%.

<b>Note</b>: Variance measured with `np.bincount(y_train)`, with `y_train` being the array of labels for the training data set.

With a 94.9% accuracy on the validation set and a 94% accuracy on the test set, this model appears to not be drastically overfitting the training data.

##### Activation Function

Rectified Linear Unit's (ReLU's) are used for the activation functions because it is less computationally complex than other activation functions, such as the scaled hyperbolic tangent function.

##### Future Considerations

1. Image Generation from Labeled Coordinates

A future modification to the preprocessing step involves utilizing the 'coords' data that specifies where a traffic sign is found within an image. With these coordinates, one can decide to initialize the weights to reflect this priotization instead of starting the weights at random or perhaps one can autogenerate more images from the road signs found in these images. I attempted to utilize the coordinates in generating new data, and I was unable to make use of them since they were not also scaled with the data that they came with.

2. Architectural Changes

Per the recommendations outlined in the [paper](https://arxiv.org/pdf/1802.06205.pdf) on SimpNet, I attempted to implement SAF's; I found that this did not improve my accuracy. However, a future modification to the current architecture would be to follow the papers suggestions and replace any 5x5 convolusional layers with multiple 3x3 layers in order to make our network 'deeper.' 

Another, more intuitive, architectural change involves not converting to grayscale and leveraging the 3 color channels that each image comes with. Although this may provide more information for differentiating traffic signs of different base colors, I believe this has the potential to introduce some noise into the model. For example, stop sign images used to train the model could be captured in a different lighting than some stop signs we wish to detect in the real world. Depending on how much weight the model put on color, the model could possibly classify the input stop sign image as something else entirely! To guard against this, generating new images with a varying amount of brightness could be mandatory in order to make the model robust in these situations.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Five online German traffic sign images are shown below with and without image preprocessing.

![onlineTrafficSignImagesBeforeAndAfterPreprocessing][onlineTrafficSignImagesBeforeAndAfterPreprocessing]

A table discussing the the qualitie(s) that may prove difficult to classify for each image is shown below.

| Label  | Possible Complications |
|:------:|:-----------:|
| Stop | The traffic sign is at an angle. Noise in background caused by tree. |
| Speed limit (80km/h) | The traffic sign is at an angle. Noise in background caused by road and mountains.  |
| Go straight or left | There is  noise caused by artificial markings on image. |
| Children Crossing | The traffic sign is flipped vertically in comparison to the children crossing signs found in the training set |
| Speed limit (50km/h)  | The image appears completely artificial and may not have some real world features that are model could be biased towards. |

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

A table showing the model's predictions for the 5 traffic sign images found online is shown below.

| Label | Prediction |
|:------:|:-----------:|
| Stop | Speed limit (30km/h) |
| Speed limit (80km/h) | Speed limit (100km/h) |
| Go straight or left | Go straight or left |
| Children crossing | Right-of-way at the next intersection |
| Speed limit (50km/h) | Speed limit (50km/h) |

The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%, thereby performing over 50% worse than both the validation set and test set. 

I believe the model incorrectly labeled the Speed limit (80km/h) image due to the image losing important traffic sign information after preprocessing. One can see above (Question 1 of this section) that the this traffic sign is unrecognizable to the human eye as an 80km/h speed limit sign after image preprocessing.

This data loss occured when reshaping the input image to a 32x32 image. Possible rectifications to this issue include correctly downsampling the image or possibly running a shape detection algorithm to automatically detect the sign and crop that section of the image.

The children crossing sign was incorrectly labeled, and the sign that it was mistakenly classified for does have the same triangular shape as the children crossing sign. The reason why the model did not label this sign correctly is likely due to this sign being the vertical inverse of all children crossing signs used in the training and validation image sets.

The stop sign was likely misidentified as a speed limit sign due to the way in which the image was reshaped to a 32x32 image. Looking at the preprocessed version of the stop sign above, one can see that the octogonal shape that makes the sign so unique is less pronounced in the preprocessed image (allowing it to be mistaken for a speed limit sign with a circular shape). 

The fact that the model misidentified the stop sign based on the shape tells me that the model places a heavier weight on shapes of traffic signs than the text/pictograms depicted within.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The model appeared to be nearly 100% certain when predicting all of the images. The original images are displayed alongside the model's top 5 predictions for each below.

![onlineTrafficSignImagesPredictions][onlineTrafficSignImagesPredictions]

Notice how the only image in which the model was not 100% certain of it's prediction, was the stop sign. The model has a 0.1% confidence in the stop sign being a stop sign, a 'close' second best!

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
