# -*- coding: utf-8 -*-
import numpy as np
from math import floor
import threading

from settings import Settings

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self): # Py3
        with self.lock:
            return self.it.next()
    
    def __len__(self):
        return Settings.BATCH_SIZE
    
    

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

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
        self._data = np.load(Settings.EMBEDDINGS_FILENAME)
            
    def __init__(self):
        self._data = []
        
    @staticmethod
    def generate(dim, n, filename):
        data = np.random.uniform(size=(n, dim), low=0, high=1)
        np.save(Settings.EMBEDDINGS_FILENAME, data)

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

    def print_vars(self, vars_r):
        vars_r = vars_r.reshape(
                (
                    Settings.N_SLOTS, 
                    Settings.SEM_DIM + Settings.MAX_ARITY * Settings.N_SLOTS))
        
        vars_sem = vars_r[:,:Settings.SEM_DIM]
        vars_struct = vars_r[:,Settings.SEM_DIM:].reshape(
                (
                    Settings.MAX_ARITY,
                    Settings.N_SLOTS,                    
                    Settings.N_SLOTS))
        
        for i in range(Settings.N_SLOTS):
            print("{} ... {}\t\t{}".format(
                        " ".join(map(lambda x: "{:>5.2f}".format(x), vars_sem[i,:5])),
                        " ".join(map(lambda x: "{:>5.2f}".format(x), vars_sem[i,-5:])),
                        "\t".join(
                            [" ".join(
                                map(
                                    lambda x: "{:>5.0f}".format(x), 
                                    vars_struct[arg, i])) 
                                for arg in range(Settings.MAX_ARITY)])
                    ))
            print("\n")
        
    def gen_random(self, n = 1):
        vars_data = np.zeros(
                shape=(
                        n,
                        Settings.N_SLOTS, 
                        Settings.SEM_DIM 
                        + Settings.MAX_ARITY * Settings.N_SLOTS))
        
        vars_data[:,:,:Settings.SEM_DIM] = np.reshape(
                self._embeddings.get_random(n=Settings.N_SLOTS * n)[0],
                (n, Settings.N_SLOTS, Settings.SEM_DIM))
        
        vars_data[:,:,Settings.SEM_DIM:] = np.random.choice(
                    [-1, 1],
                    p=[0.8, 0.2],
                    size=(
                        n, 
                        Settings.N_SLOTS,
                        Settings.MAX_ARITY * Settings.N_SLOTS)
                )
        
        return vars_data.reshape((n, Settings.VARS_TOTAL_DIM))

#    @threadsafe_generator
    def get_generator(self, batch_size, fixed_target = False):
        sim_targets = np.ones(shape=(batch_size), dtype=np.float32)

        vars_input = np.zeros(
                shape=(
                    batch_size,
                    Settings.VARS_TOTAL_DIM,
                    2),
                dtype=np.float32)

        while(True):

            with self.lock:
                vars_input[:,:,1] = self.gen_random(n=batch_size)
                if fixed_target:
                    vars_input[:,:,0] = self.gen_random(n=1)
                else:
                    vars_input[:,:,0] = self.gen_random(n=batch_size)
                
            
                yield vars_input, sim_targets
            
    def __init__(self):
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
        self.lock = threading.Lock()
        
#vars_data = VARSData()
##data = next(vars_data.get_generator(1, fixed_target=True))
##vars_data.print_vars(data[0][0,:,0])
#sems = []
#structs = []
#for _ in range(10000):
#    data = vars_data.gen_random(2).reshape(
#            2, 
#            Settings.N_SLOTS, 
#            Settings.SEM_DIM + Settings.MAX_ARITY * Settings.N_SLOTS)
#    import tensorflow as tf
#    a_sem = data[0,:,:Settings.SEM_DIM].flatten()
#    b_sem = data[1,:,:Settings.SEM_DIM].flatten()
#    a_struct = data[0,:,Settings.SEM_DIM:].flatten()
#    b_struct = data[1,:,Settings.SEM_DIM:].flatten()
#    sems.append(tf.norm(a_sem - b_sem).numpy())
#    structs.append(tf.norm(a_struct - b_struct).numpy())
#    
#print((np.mean(sems), np.std(sems)))
#print((np.mean(structs), np.std(structs)))

    