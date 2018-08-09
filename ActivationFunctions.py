import abc
import numpy as np


class ActivationFunction(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self):
        pass

    @abc.abstractmethod
    def der(self):
        pass

    @property
    def value(self):
        return self.__value

    @value.setter
    def value_setter(self, newValue):
        self.__value = newValue


class sigmoid(ActivationFunction):
    def __call__(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("data should be in the form of an numpy array")
        if data.ndim != 2:
            raise ValueError("data should be 2D")

        self.value_setter = 1/(1-np.exp(-data))
        return self.value

    @property
    def value(self):
        return self.__value

    @value.setter
    def value_setter(self, newValue):
        self.__value = newValue

    def der(self):
        return self.value*(1-self.value)
