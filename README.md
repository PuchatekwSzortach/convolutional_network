# convolutional_network

A simple convolutional network library for image classification. Following layers are made available:

- net.layers.Input
- net.layers.Convolution2D
- net.layers.Flatten
- net.layers.Softmax

Use it as in code below:

```python
import net.layers
import net.model

layers = [
        net.layers.Input(sample_shape=(28, 28, 1)),
        net.layers.Convolution2D(filters=20, rows=14, columns=14),
        net.layers.Convolution2D(filters=10, rows=15, columns=15),
        net.layers.Flatten(),
        net.layers.Softmax()
    ]

model = net.models.Model(layers)

for x_batch, y_batch in x_train, y_train:
    model.train(x_batch, y_batch, learning_rate=0.01)

print("Accuracy: {}".format(model.get_accuracy(x_test, y_test)))
```

A more comprehensive example can be found in `./examples/mnist.py`.

Only 2D convolutions are available and each convolution applies ReLU activation to its output. Padding is not supported, all convolutions are performed in `valid` mode only. Output of last convolutional layer has to have spatial size 1x1 and channels count equal to number of classes you want to predict. Final two layers must be `net.layers.Flatten` followed by `net.layers.Softma`. Simple stochastic gradient descent is used to train networks parameters.

### Dependencies ###

The only dependencies library itself has are `python 3` and `numpy`.   
In addition `./examples/mnist.py` script uses `sklearn` to load MNIST data and `tqdm` for progress bar.

### Performance ###
While this work is no match for commercial grade deep learning libraries, a reasonable amount of effort has been put into optimizing convolutions. All critical computations are performed with matrix dot products.
