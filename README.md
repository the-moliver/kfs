# Keras for Science (KFS)

Keras for Science (KFS) is a set of extensions to the excellent [Keras](https://github.com/chollet/keras) neural network library.

Most of the deep learning literature has focused on classification problems, while many problems in science are actually regression problems. Also scientific data is often significantly smaller (in terms of number of samples) and more noisy than the data typically used in deep learning. I've been using deep learning for several years now to model data recorded from neurons in visual cortex. I created these extensions to Keras to make the lessons I've learned applying deep learning to scientific problems easily accessible.

Time series regression problems are common in science so KFS includes a model type that makes it simple to incorporate lagged input values into the model. Also included are layer types that have applications in neuroscience and beyond: temporal filtering, non-negative projections, flexible divisive normalization. KFS was designed to be used along with Keras, importing features from KFS as needed.

To demonstrate its proper use I've also included a few examples of KFS applied to real data.

------------------



## Getting started: 30 seconds to Keras for Science

The core data structure of Keras is a __model__, a way to organize layers. KFS adds time delays to the standard [`Sequential`](http://keras.io/models/#sequential) model in Keras.

Here's the `time_delay_generator` which automatically generates time delays (i.e. past values of the input):

```python
from kfs.generators import time_delay_generator

time_delays = 10
batch_size = 128
train_gen = time_delay_generator(X_train, Y_train, delays, batch_size)

```

Layers from Keras can then be imported and used

```python
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Flatten, Activation

model.add(TimeDistributed(Dense(output_dim=64), input_shape=(time_delays, 100,)))
model.add(Flatten())
model.add(Dense(output_dim=64))
model.add(Activation("relu"))
model.add(Dense(output_dim=1))
model.add(Activation("relu"))

```

Once your model is complete, it can be compiled and fit:
```python
from keras.optimizers import SGD
model.compile(loss='poisson', optimizer=SGD(lr=0.0001, momentum=0.5, nesterov=True))
model.fit_generator(train_gen, samples_per_epoch=X_train.shape[0], nb_epoch=100)
```

Once fit you can test the model by predicting on held out data:
```python
tst_gen = time_delay_generator(X_test, None, delays, batch_size, shuffle=False)
pred = model1.predict_generator(tst_gen, X_test.shape[0])
```
Check the [examples folder](https://github.com/the-moliver/kfs/tree/master/examples) of the repo for more examples


------------------
