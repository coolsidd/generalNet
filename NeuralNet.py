import numpy as np
import ActivationFunctions


class NeuralNet:
    def __init__(self, inputDims):
        self.__inputDims = 0
        self.minibatch_size = None
        self.__layers = []
        self.__inputDims = inputDims

    @property
    def inputDims(self):
        return self.__inputDims

    @inputDims.setter
    def setInputDims(self, dims):
        if isinstance(dims, tuple) or isinstance(dims, list) or isinstance(dims, int):
            if len(dims) <= 2:
                self.__inputDims = dims[0]
                if len(dims) == 2:
                    self.minibatch_size = dims[1]
                    temp = 0
                    # for layer in self.layers:
                    #     temp = layer.nodes
                    # write code to modify all input layers to the correct dims
            else:
                raise ValueError(
                    "currently only inputs of upto 2 dimensions supported")
        else:
            raise TypeError("dims must be of type list of tuple or int")

    @property
    def layers(self):
        return self.__layers

    def add_layer(self, nodes, activationFunction=None):
        if activationFunction is None:
            activationFunction = ActivationFunctions.sigmoid()
        if not isinstance(nodes, int):
            raise TypeError("Number of nodes must be an integer")
        if self.inputDims is None:
            raise NotImplementedError
        if len(self.layers) == 0:
            numNodesInPrevLayer = self.inputDims
        else:
            numNodesInPrevLayer = self.layers[-1].numNodes

        self.layers.append(
            Layer(activationFunction, nodes, numNodesInPrevLayer))

    def forward_prop(self, data):
        if data.shape[0] != self.inputDims:
            raise ValueError("Number of inputs does not match\nExpected {} found {}".format(
                self.inputDims, data.shape[0]))
        for Layer in self.layers:
            data = Layer.forward_prop(data)
        return data


class Layer:
    def __init__(self, actFunc, numNodes, numNodesInPrevLayer):
        self.activationFunction = actFunc
        self.__numNodes = numNodes
        self.__numNodesInPrevLayer = numNodesInPrevLayer
        self.weights = np.random.random((numNodes, numNodesInPrevLayer))*2 - 1
        self.bias = np.random.random((numNodes, 1))*2 - 1

    def forward_prop(self, data):
        if not isinstance(data, np.ndarray):
            raise("Data must be an numpy array")
        self.value = self.weights.dot(data) + self.bias
        return self.activationFunction(self.value)

    @property
    def numNodes(self):
        return self.__numNodes

    # @numNodes.setter
    # def numNodesSetter(self, val):
    #     if not isinstance(val, int):
    #         raise TypeError("number of nodes must be an int")
    #     if not (val > 0):
    #         raise ValueError("number of nodes should be positive")

    #     if val >= self.__numNodes:
    #         temp = np.random.random(
    #             (val - self.__numNodes, self.numNodesInPrevLayer))
    #         self.weights = np.append(self.weights, temp, 1)
    #         temp = np.random.random((val-self.__numNodes, 1))
    #         self.bias = np.append(self.bias, temp, 0)

    @property
    def numNodesInPrevLayer(self):
        return self.__numNodesInPrevLayer

    # @numNodesInPrevLayer.setter
    # def numNodesInPrevLayerSetter(self, val):
    #     if not isinstance(val, int):
    #         raise TypeError("number of nodes must be an int")
    #     if not (val > 0):
    #         raise ValueError("number of nodes should be positive")

    #     if val >= self.__numNodesInPrevLayer:
    #         temp = np.random.random(
    #             (self.numNodes, val - self.__numNodesInPrevLayer))
    #         self.weights = np.append(self.weights, temp, 0)
    #         temp = np.random.random((val-self.__numNodes, 1))
    #         self.bias = np.append(self.bias, temp, 0)
