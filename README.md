# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/distribution_data_set_1.png "Data set distribution"
[image2]: ./examples/pre_processed_image.png "Pre_processed image"
[image3]: ./examples/distribution_data_set_2.png "Final  data set distribution"
[image4]: ./traffic-signs-images/1.png "Traffic Sign 1"
[image5]: ./traffic-signs-images/2.png "Traffic Sign 2"
[image6]: ./traffic-signs-images/3.png "Traffic Sign 3"
[image7]: ./traffic-signs-images/4.png "Traffic Sign 4"
[image8]: ./traffic-signs-images/5.png "Traffic Sign 5"
[image9]: ./examples/softmax_prob.png "Softmax probability"

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/mhBahrami/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples is 34799
* Number of testing examples is 12630
* Number of validation examples is 4410
* Image data shape is (32, 32, 3)
* Number of classes is 43

#### 2. Distribution of classes in the training, validation and test set

As you can see here distribution of all data sets are almost the same. Also there are more examples of some classes like "speed limit" signs than the others.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Pre-process the Data Set (normalization, grayscale, etc.)

##### Preprocessing the input images

To have a better data set to train, validate, and test the model I did the following steps:

- Gamma adjustment
- Converting the RGB image to a grayscale image
- Contrast adjustment
- Normalization: for normalization I used `(pixel - mean)/std`, in which `mean` is the mean of image and `std` is standard deviation of image

> `helper.image_preprocessing()` function is responsible to preprocess each image in data set.

**Example**

```
shape in/out: (32, 32, 3) (32, 32, 1)
```

![alt text][image2]

##### Building the jittered dataset

Additionally, I build a jittered dataset by adding 4 transformed versions of the original training set, yielding 173995 samples for training, 22050 for validation, and 63150 for testing. Samples are randomly perturbed in scale (`[0.9,1.1]` ratio) and rotation (`[-15,+15]` degrees).

You can see new data set distributions below.

![alt text][image3]

The difference between the augmented data set and the original data set is the following:

1. The color channel is GRAY.
2. The mean of each image is **0**.
3. Includes scaled and rotated images.

#### 2. Setup TensorFlow

##### Dimensionality

Given:

- our input layer has a width of `W` and a height of `H`
- our convolutional layer has a filter size `F`
- we have a stride of `S`
- a padding of `P`
- and the number of filters `K`,

the following formula gives us the width of the next layer: `W_out = [(W−F+2P)/S] + 1`.

The output height would be `H_out = [(H-F+2P)/S] + 1`.

And the output depth would be equal to the number of filters `D_out = K`.

The output volume would be `W_out * H_out * D_out`.

> `helper.get_variable_sizes()` calculates the size of all layers with 2 fully-connected layers.

**Example**

```python
_ = helper.get_variable_sizes(43, stride = 1, k_size = 2, padding = 'VALID',\
                       w_in = 32, h_in = 32, d_in = 1, \
                       f0_size=5, f1_size=5, f1_d=6, l2_d=16, \
                       fc1_out = 800, fc2_out= 200)
```

**Result**

- filter0: 5x5x1
- conv1: 28x28x6
- max_p1: 14x14x6
- filter1: 5x5x6
- conv2: 10x10x16
- max_p2: 5x5x16
- fc1_in: 400
- fc1_out: 800
- fc2_in: 800
- fc2_out: 200
- out_in: 200
- out_size: 43

> `helper.get_variable_sizes2()` calculates the size of all layers with only 1 fully-connected layer.

**Example**

```python
_ = helper.get_variable_sizes2(43, stride = 1, k_size = 2, padding = 'VALID',\
                       w_in = 32, h_in = 32, d_in = 1, \
                       f0_size=5, f1_size=5, f1_d=6, l2_d=16, \
                       fc1_out = 800)
```

**Result**

- filter0: 5x5x1
- conv1: 28x28x6
- max_p1: 14x14x6
- filter1: 5x5x6
- conv2: 10x10x16
- max_p2: 5x5x16
- fc1_in: 400
- fc1_out: 800
- out_in: 800
- out_size: 43

#### Final Model

##### Input

The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels. C is 1 in this case. Because the processed images are gray scale.

##### Architecture

- **Layer 1: Convolutional.** The output shape should be 25x25x32.
  -  1x1 stride, `VALID` padding
- **Activation.** ReLU is the activation function.
- **Pooling.** The output shape should be 12x12x32.
  - 2x2 stride, `VALID` padding
- **Layer 2: Convolutional.** The output shape should be 8x8x64.
  -  1x1 stride, `VALID` padding
- **Activation.** ReLU is the activation function.
- **Pooling.** The output shape should be 4x4x64.
  - 2x2 stride, `VALID` padding
- **Flatten.** Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using `tf.contrib.layers.flatten`, which is already imported. In this model the output size is 1024.
- **Layer 3: Fully Connected.** This should have 512 outputs.
- **Activation.** ReLU is the activation function.
  - *Drop out layer.* Only for training with `keep_prob=0.5`.
- **Layer 4: Fully Connected.** This should have 128 outputs.
- **Activation.** ReLU is the activation function.
  - *Drop out layer.* Only for training with `keep_prob=0.5`.
- **Layer 5: Fully Connected (Logits).** This should have 43 outputs.
- **Output** Return the result of the 2nd fully connected layer.

#### 3. Train, Validate and Test the Model

I used `tf.nn.softmax_cross_entropy_with_logits()` to calculate the loss. Also I used `tf.train.AdamOptimizer()` as optimizer to minimize the loss. 

##### Changing training model and hyper parameters

With *changing the size of each layer or hyper parameters* we can build and train a model that will be able to predict the results with a high accuracy. During tuning the training sometimes I reduced the `learning_rate` to avoid divergence and/or big oscillation in accuracy. Consequently, I increased the `epochs` due to a slow convergence. Also, we shouldn't choose very big or small `batch_size`. Something like 100 is good enough for training this model.

You can see the first 5 best results below. **The last one is the final trained model.**  

##### 1. Validation Accuracy: 0.955

------

- `epochs = 25`
- `batch_size = 100`
- `learning_rate = 0.0009`
- `kp = 0.50`
- `strides = 1`
- `kernels = 2`
- `padding = 'VALID'`
- For weights and biases:
  - `mu = 0`, `sigma = 0.0001`
- Others:
  - `n_input_depth = 1, w_in = 32, h_in = 32, f0_size=5, f1_size=5, f1_d=6, l2_d=16, fc1_out = 800, fc2_out= 200`
- 2 fully connected layout before output layout

##### 2. Validation Accuracy: 0.950

------

- `epochs = 25`
- `batch_size = 100`
- `learning_rate = 0.0009`
- `kp = 0.50`
- `strides = 1`
- `kernels = 2`
- `padding = 'VALID'`
- For weights and biases:
  - `mu = 0`, `sigma = 0.0001`
- Others:
  - `n_input_depth = 1, w_in = 32, h_in = 32, f0_size=5, f1_size=5, f1_d=6, l2_d=16, fc1_out = 800`
- 1 fully connected layout before output layout

##### 3. Validation Accuracy: 0.941

------

- `epochs = 25`
- `batch_size = 100`
- `learning_rate = 0.0009`
- `kp = 0.50`
- `strides = 1`
- `kernels = 2`
- `padding = 'VALID'`
- For weights and biases:
  - `mu = 0`, `sigma = 0.0001`
- Others:
  - `n_input_depth = 1, w_in = 32, h_in = 32, f0_size=5, f1_size=5, f1_d=6, l2_d=16, fc1_out = 800, fc2_out= 200`
- 2 fully connected layout before output layout and new data set

##### 4. Validation Accuracy: 0.965

------

- `epochs = 25`
- `batch_size = 128`
- `learning_rate = 0.001`
- `kp = 0.50`
- `strides = 1`
- `kernels = 2`
- `padding = 'VALID'`
- For weights and biases:
  - `mu = 0`, `sigma = 0.0001`
- Others:
  - `n_input_depth = 1, w_in = 32, h_in = 32, f0_size=8, f1_size=5, f1_d=32, l2_d=64, fc1_out = 512, fc2_out= 128`
- 2 fully connected layout before output layout and new data set

##### 5. Validation Accuracy: 0.967 (Selected as Final Model)

------

- `epochs = 40`
- `batch_size = 128`
- `learning_rate = 0.0008`
- `kp = 0.50`
- `strides = 1`
- `kernels = 2`
- `padding = 'VALID'`
- For weights and biases:
  - `mu = 0`, `sigma = 0.0001`
- Others:
  - `n_input_depth = 1, w_in = 32, h_in = 32, f0_size=8, f1_size=5, f1_d=32, l2_d=64, fc1_out = 512, fc2_out= 128`
- 2 fully connected layout before output layout and new data set.



##### Test Model

Test your model against the test dataset. This will be your final accuracy. As you can see the accuracy of the model is 94%.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8]

The second image might be difficult to classify because it is like the signs for 50 or 80 (km/h) speed limit.

#### 2. The model's predictions on these new traffic signs

Here are the results of the prediction:

|                 Image                 |              Prediction               |
| :-----------------------------------: | :-----------------------------------: |
| Right-of-way at the next intersection | Right-of-way at the next intersection |
|         Speed limit (30km/h)          |         Speed limit (30km/h)          |
|             Priority road             |             Priority road             |
|            Turn left ahead            |            Turn left ahead            |
|            General caution            |            General caution            |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Output Top 5 Softmax Probabilities For Each Image Found on the Web

##### Print out the top five Softmax probabilities for the predictions on the German traffic sign images found on the we

For second image, the model is relatively sure that this is a "Speed limit (30km/h)" sign (probability of 0.78), and the image does contain it. And for the rest of images, the model is 100% sure about the type of the sign. The top five soft max probabilities for the images were

**Right-of-way at the next intersection**

```python
[ 9.99286234e-01, 7.13804853e-04, 1.18973081e-10, 2.27878862e-11, 1.23311396e-11 ]
```

**Speed limit (30km/h)**

```python
[ 7.84267783e-01, 2.14224666e-01, 1.49137375e-03, 6.61399281e-06, 6.19813272e-06 ]
```

**Priority road**

```python
[ 1.00000000e+00, 1.90708209e-19, 3.26064036e-21, 8.25148961e-22, 2.53944625e-22 ]
```

**Turn left ahead**

```python
[ 1.00000000e+00, 2.75898139e-16, 3.73007212e-18, 2.15126768e-20, 1.12354699e-20 ]
```

**General caution**

```python
[ 1.00000000e+00, 5.83028209e-11, 6.30079222e-15, 2.54975032e-18, 1.11121951e-21 ]
```

| Probability |              Prediction               |
| :---------: | :-----------------------------------: |
|    1.00     | Right-of-way at the next intersection |
|    0.78     |         Speed limit (30km/h)          |
|    1.00     |             Priority road             |
|    1.00     |            Turn left ahead            |
|    1.00     |            General caution            |

You can see the result of Softmax probability for the images below.

![alt text][image9]