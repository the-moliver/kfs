# Keras for Science (KFS)

Keras for Science (KFS) is a set of extensions to the excellent [Keras](https://github.com/chollet/keras) neural network library.

Most of the deep learning literature has focused on classification problems, while many problems in science are actually regression problems. Also scientific data is often significantly smaller (in terms of number of samples) and more noisy than the data typically used in deep learning. I've been using deep learning for several years now to model data recorded from neurons in visual cortex. I created these extensions to Keras to make the lessons I've learned applying deep learning to scientific problems easily accessible.

Time series regression problems are common in science so KFS includes a model type that makes it simple to incorporate lagged input values into the model. Also included are layer types that have applications in neuroscience and beyond: temporal filtering, non-negative projections, flexible divisive normalization. KFS was designed to be used along with Keras, importing features from KFS as needed.

To demonstrate its proper use I've also included a few examples of KFS applied to real data.

------------------



## Getting started: 30 seconds to Keras for Science

The core data structure of Keras is a __model__, a way to organize layers. KFS adds time delays to the standard [`Sequential`](http://keras.io/models/#sequential) model in Keras.

Here's the `Sequential` model with time delays (i.e. past values of the input):

```python
from kfs.tdmodels import Sequential

time_delays = 10
model = Sequential(delays=time_delays)
```

Layers from Keras can then be imported and used. Note the input shape includes ```time_delays+1``` because it includes the current time in addition to ```time_delays```

```python
from keras.layers.core import TimeDistributedDense, Dense, Flatten, Activation

model.add(TimeDistributedDense(output_dim=64, input_shape=(time_delays+1, 100,))
model.add(Flatten())
model.add(Dense(output_dim=64))
model.add(Activation("relu"))
model.add(Dense(output_dim=1))

```

Once your model is complete, it can be compiled and fit:
```python
from keras.optimizers import SGD
model.compile(loss='poisson', optimizer=SGD(lr=0.0001, momentum=0.5, nesterov=True))
model.fit(X_train, Y_train, nb_epoch=500, batch_size=512)
```

Or generate predictions on new data:
```python
pred = model.predict(X_test)
```
Check the [examples folder](https://github.com/the-moliver/kfs/tree/master/examples) of the repo for more examples


------------------
