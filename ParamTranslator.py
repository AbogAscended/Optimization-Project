from torch import Tensor as T
import numpy as np
from typing import Iterable

class ParamTranslator:
    def __init__(self, params: Iterable[T] = None):
        assert isinstance(params, Iterable[T]), 'Needs to be iterable parameters of neural network'
        self.params = params
        self.tensors = []
    
    def getTensors(self, params: Iterable[T] = None):
        assert isinstance(params, Iterable[T]), 'Needs to be iterable parameters of neural network'
        if params is not None: self.params = params
        for param in self.params:
            self.tensors.append(T.numpy(param.clone().detach()))
    
    def saveTensors(self, filepath):
        for i, matrix in enumerate(self.tensors):
            np.save(f'{filepath}{i+1}.npy', matrix)

    def updateTensor(self, paramMat: np.array = None, paramMatIDX = None, allParams: list = None):
        if paramMat is not None and paramMatIDX is not None:
            self.tensors[paramMatIDX] = paramMat
        elif paramMat is not None and paramMatIDX is None:
            return f'Error, no index passed as parameter for matrix'
        elif paramMat is None and paramMatIDX is not None:
            return f'Error, index passed but no matrix passed'
        if paramMat is None and paramMatIDX is None and allParams is not None:
            assert all(isinstance(param, np.ndarray) for param in allParams), 'List does not contain all numpy arrays'
            self.tensors = allParams
    
    def getParams(self,vec:list, params: Iterable[T]):
        assert isinstance(params, Iterable[T]), 'Needs to be iterable parameters of neural network'
        for idx, param in enumerate(params):
            param.data = vec[idx]