# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os
from math import floor

from settings import Settings

class EmbeddingsData:
    def get_random(self, n = 1, exclude_indices = [], region = None):
        if region is None:
            region = len(self._data)
            
        while True:
            indices = np.random.choice(region, n, replace=True)
            if set(indices).intersection(set(exclude_indices)) == set():
                return self._data[indices], indices
            
    def get(self, index):
        return self._data[index]
    
    def load(self, filename):
        with open(filename, "rb") as F:
            self._data = pickle.load(F)
            
    def __init__(self):
        self._data = []
        
    @staticmethod
    def generate(dim, n, filename):
        data = np.random.uniform(size=(n, dim), low=0, high=1)
        with open(filename, "wb") as F:
            pickle.dump(data, F)

class CategoryEmbeddings(EmbeddingsData):
    
    def get_random_from_cat(self, cat, n = 1, exclude_indices = []):
        assert(cat in self.__cats)
        
        region = list(range(
                floor(self.__cats[cat][0] * len(self._data)), 
                floor(self.__cats[cat][1] * len(self._data))))
            
        return self.get_random(
                n=n, 
                exclude_indices=exclude_indices,
                region=region)
    
    def __init__(self, cats):
        super().__init__()
        self.__cats = cats
        
class VARSData:
    
    def __init__(self):
        if not os.path.exists(Settings.EMBEDDINGS_FILENAME) :
            EmbeddingsData.generate(
                    dim=Settings.SEM_DIM, 
                    n=Settings.N_SYMBOLS,
                    filename=Settings.EMBEDDINGS_FILENAME)        
        
        cats = {
                "atom": (0, Settings.ATOM_SYMBOLS_FREQ),
                "unary": (
                        Settings.ATOM_SYMBOLS_FREQ, 
                        Settings.ATOM_SYMBOLS_FREQ 
                            + Settings.UNARY_SYMBOLS_FREQ),
                "binary": (
                        Settings.ATOM_SYMBOLS_FREQ 
                            + Settings.UNARY_SYMBOLS_FREQ,
                        Settings.ATOM_SYMBOLS_FREQ 
                            + Settings.UNARY_SYMBOLS_FREQ,
                            + Settings.BINARY_SYMBOLS_FREQ
                        )
            }
        self._embeddings = CategoryEmbeddings(cats = cats)
        self._embeddings.load(Settings.EMBEDDINGS_FILENAME)
)
        
vars_data = VARSData()
    
    
    