# convolutional_network
Nothing much in here yet, please come back later.

Ok, think this through.

We will have layers:
Input, Convolution2D, Flatten, Softmax

And model class.

We will make model with:
Model(layers)

It will check that it starts with Input and finished with Softmax

Softmax will make checks, etc, etc.

Model will have api:
- compile
- fit
- predict

Model.compile:

    - model checks first layer is input and last is softmax

    Then starting from first to last:
    - for Input - set output_shape
    - for Convolution2D:
        - get depth of its predecessor
        - initialize kernels accordingly
        - set its own output_shape
    - for Flatten - call np.flat(), make sure output is 2D, set output_shape
    - Softmax - make sure output is 2D and labels dimension is 2 or more

Model.predict:

    1. For each layer call forward
        - Input.forward(x) - checks shapes correctness
        - Convolution2D.forward(x) - gets data, convolves, adds biase, relu nonlinearity, outputs result
        - Flatten.forward(x) - calls np.flat, makes sure output is 2D, outputs result
        - Softmax - computes softmax, outputs result - makes sure input to softmax has expected shape
    2. Output result

Model.fit:

    Shuffle data
    For each batch:

       - For each layer from start to end:

            Call train_forward() - does computations as in forward, but caches inputs and outputs

        - For each layer from end to start:

            Call train_backward() - performs backpropagation and applies results

