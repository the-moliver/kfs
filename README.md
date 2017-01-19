# Keras for Science (KFS)

Keras for Science (KFS) is a set of extensions to the excellent [Keras](https://github.com/fchollet/keras) neural network library.

Most of the deep learning literature has focused on classification problems, while many problems in science are actually regression problems. Also scientific data is often significantly smaller (in terms of number of samples) and more noisy than the data typically used in deep learning. I've been using deep learning for several years now to model data recorded from neurons in visual cortex. I created these extensions to Keras to make the lessons I've learned applying deep learning to scientific problems easily accessible.

Time series regression problems are common in science so KFS includes a model type that makes it simple to incorporate lagged input values into the model. Also included are layer types that have applications in neuroscience and beyond: temporal filtering, non-negative projections, flexible divisive normalization. KFS was designed to be used along with Keras, importing features from KFS as needed.

To demonstrate its proper use I've also included a few examples of KFS applied to real data.

------------------



## Getting started: 30 seconds to Keras for Science

KFS adds the ability to use arbitrary time delays with the standard [`Sequential`](http://keras.io/models/#sequential) model in Keras. This is accomplished by the use of a generator which creates batches of stimuli of shape `(batch_size, delays, input_dim)` from a time series data matrix of shape `(samples, input_dim)`. Each batch will consist of multiple (i.e. `batch_size`) randomly sampled continuous windows (of length `delays`) of the time series data. This is useful for creating models that operate on windows of continuous data such as Recurrent Neural Networks or Convolutional Neural Networks.

Here's the `time_delay_generator` which automatically generates batches of time series data:

```python
from kfs.generators import time_delay_generator

delays = 10
batch_size = 128
train_gen = time_delay_generator(X_train, Y_train, delays, batch_size)
```

Layers from Keras can then be imported and used:
```python
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Flatten, Activation

model.add(TimeDistributed(Dense(output_dim=64), input_shape=(delays, 100,)))
model.add(Flatten())
model.add(Dense(output_dim=64))
model.add(Activation("relu"))
 This model.add(Dense(output_dim=1))
*model.add(Activation("relu"))
```

Once your model is complete, it can be compiled and fit:
```python
from keras.optimizers import SGD
model.compile(loss='poisson', optimizer=SGD(lr=0.0001, momentum=0.5, nesterov=True))
model.fit_generator(train_gen, samples_per_epoch=X_train.shape[0], nb_epoch=100)
```

Once fit you can test the model by predicting on held out data. Set `shuffle` to `False` so the samples are generated in the correct order:
```python
tst_gen = time_delay_generator(X_test, None, delays, batch_size, shuffle=False)
pred = model.predict_generator(tst_gen, X_test.shape[0])
```
Check the [examples folder](https://github.com/the-moliver/kfs/tree/master/examples) of the repo for more examples


------------------

### Applying weights to arbitrary axes
* `FilterDims` The layer lets you filter any arbitrary set of axes by projection onto a new axis. This function is very flexible and incredibly useful for reducing dimensionality and/or regularizing spatio-temporal models or other models of structured data.

For example, in a 5D spatio-temporal model with input shape (#samples, 12, 3, 30, 30) the input has 12 time steps, 3 color channels and X and Y of size 30:
```python
model = Sequential()
model.add(TimeDistributed(Convolution2D(10, 5, 5, activation='linear', subsample=(2, 2)), input_shape=(12, 3, 30, 30)))
```
The output from the previous layer has shape (#samples, 12, 10, 13, 13). We can use FilterDims to filter the 12 time steps on axis 1 by projeciton onto a new axis of 5 dimensions with a 12x5 matrix:
```python
model.add(FilterDims(nb_filters=5, sum_axes=[1], filter_axes=[1], bias=False))
```        
The weights learned by FilterDims are a set of temporal filters on the output of the spatial convolutions. The output dimensionality is (#samples, 5, 10, 13, 13). We can then use FilterDims to filter the 5 temporal dimensions and 10 convolutional filter feature map dimensions to create 2 spatio-temporal filters with a 5x10x2 weight tensor:
.```python
model.add(FilterDims(nb_filters=2, sum_axes=[1, 2], filter_axes=[1, 2], bias=False))
``` 
The output dimensionality is (#samples, 2, 13, 13) since we have collapsed dimensions [1, 2]. We can then use FilterDims to separately spatially filter the output of each spatio-temporal filter with a 2x13x13 tensor:
```python
model.add(FilterDims(nb_filters=1, sum_axes=[2, 3], filter_axes=[1, 2, 3], bias=False))
```
We only sum over the last two spatial axes resutling in an output dimensionality of (#samples, 2).

### Additional Regularization Options
* `TVRegularizer` Total-Variation (TV) and Total-Variation of the gradient (TV2) regularization applied along specific axes of the weight tensor. This is useful for enforcing smoothness, while maintaining edges, in spatial filters.
* `LaplacianRegularizer` Regularizer for L1 and L2 Norm of the Laplacian operator applied along a specific axis of the weight tensor. This is useful for enforcing smoothness along 1 dimension of a weight matrix.
* `XCovRegularizer` Cross-Covariance Regularization of hidden unit activity, useful for disentagling factors by penalizing similar representations.
* `StochasticWeightRegularizer` Regularization to encourage a weight tensor to sum to 1 along a specified axis.

### Additional Constraints
* `UnitNormOrthogonal` Constrain pairs of weights to be orthogonal and to have unit norm. Use
* `Stochastic` Constrain the weights incident to be positive and sum to one along a specified axis.

### Additional Noise Layers
* `CoupledGaussianDropout` Adds Gaussian noise to the output of hidden units whose variance is equal to the output. This differs from `GaussianDropout` where the Gaussian noise has variance equal to the square of the output.
* `ShapeDropout` Allows specification of axes to apply Dropout. Similar to `SpatialDropout1D`, etc. but more flexible.
* `Gain` Multiplies the outputs of a hidden layer by a small gain factor. This encourages larger weights in the layers preceding that `Gain` layer.

### Convolutional Energy Layers
* `Convolution2DEnergy` Convolution operator for filtering inputs with learned filters inspired by simple-cell and complex-cell V1 neurons.
* `Convolution2DEnergy_Scatter` Convolution operator for filtering a stack of two-dimensional inputs with learned filters inspired by simple-cell and complex-cell V1 neurons. Applies 2D filters separately to each element of the input stack, so the number of feature maps output grows multiplicatively.
* `Convolution2DEnergy_TemporalBasis` Convolution operator for filtering windows of time varying two-dimensional inputs, such as a series of movie frames, with learned filters inspired by simple-cell and complex-cell V1 neurons. Filters are learned in a factorized representation, consisting of orthogonal 2D filters, a set of vectors that control filter amplitude over time, and a set of scalars that control the trade-off of the orthogonal 2D filters over time. This representation can create a large number of 3D spatio-temporal filters from a small number of parameters, often with less than 1% of the parameters of a naive 3D convolutional model. Useful as the first layer of a network.

### Additional Dense Layers
* `SoftMinMax` Computes a selective and adaptive soft-min or soft-max over the inputs. Each output can select how it weights the inputs and the strength of the soft-min/max operation which is controlled by a parameter.
* `DenseNonNeg` A densely-connected NN layer with weights soft-rectified before being applied. Useful for NMF type applications. 
* `DivisiveNormalization` A layer where the output consists of the inputs divided by a weighted combination of the inputs.
* `Feedback` A layer where the output consists of the inputs added to a weighted combination of the inputs.

### Optimizer with Gradient Accumulation
* `NadamAccum` Nadam optimizer which can accumulate gradients across minibatches before updating model parameters.

### Additional Error Functions
* `logcosh_error` A cost function for robust regression that behaves like squared error near 0 and absolute error further away, similar to the Huber loss function
