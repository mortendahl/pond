# Pond

Experimental library for training and evaluating deep learning networks on encrypted data, especially CNNs. As such, the exposure of training data, model weights, and prediction inputs can be tightly controlled and remain private if needed.

Essentially an efficient implementation of the techniques outlined [here](http://mortendahl.github.io/2017/09/19/private-image-analysis-with-mpc/): a specialisation of the SPDZ protocol to the two-party setting with passive security and a trusted crypto producer, and optimised for the operations needed in deep learning.


## Road map

### 1. Encrypted tensors

End-to-end SPDZ implementation of basic operations on private tensors:
1. fixed-point encoding
2. sharing and reconstruction
3. scalar addition and subtraction
4. matrix dot product

Goal is to test feasibility by benchmarking matrix operations when run on e.g. two GCE instances in same data center.


### 2. Deep learning on public values

Implement layers needed for MNIST training on *public values* (testing and debugging without encryption is useful):
1. fixed-point encoding
2. Sample loading
3. Layers (forward and backwards) [needed](http://mortendahl.github.io/2017/09/19/private-image-analysis-with-mpc/)
4. SGD optimiser
5. Weight loading (from Keras)

Compare to optimised implementations such as TensorFlow to get baseline performance. Will tell us how precision is needed by fixed-point encoding (and hence the size of the field).


### 3. Deep learning on private values

Connect encrypted tensors and deep learning:
1. activations on private values
2. convolutions on private values
3. pooling on private values
4. private vector mixing

This is where we will finally get concrete performance numbers.


### 4. GPU support

Introduce support of encrypted training on GPUs:
1. Implement needed kernels for computing on integers (e.g. matrix dot product)
2. If 64bit are not enough then use CRT representation
