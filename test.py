# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

import time

from model import Model
from data import VARSData
from settings import Settings

tf.compat.v1.disable_eager_execution()
#tf.compat.v1.enable_eager_execution()


def run_rest(test):
    
#    print(test.shape)
    
    batch_size = test.shape[0]

    Settings.BATCH_SIZE = batch_size

    model = Model()
    
    def gen_data():
        while(True):
            yield test
    
    data = gen_data()
    
    d = next(data)
    d = np.float32(d)
    
    ma = model.make_analogy(d[:,:,0], d[:,:,1])
    
    
    return
#    print(tf.keras.backend.eval(ma))
    
#    return
#    exit(0)
    outputs = next(model.test(data))
    
    output_ranks = np.argsort(outputs[0]).argsort()
    vaar_ranks = np.argsort(outputs[1]).argsort()    
    
    print(outputs[1])
#    print(output_ranks)
#    print(vaar_ranks)


test1 = np.load("test2.1.npy")
test2 = np.load("test2.2.npy")

run_rest(test1)
run_rest(test2)

exit(0)

model = Model()

model.load()

start = time.time()
rank_d = []

reps = 100

varsData = VARSData()
data_gen = varsData.get_generator(
        batch_size=Settings.BATCH_SIZE,
        fixed_target=True)

for _ in range(reps):
    outputs = next(model.test(data_gen))
    
    
    output_ranks = np.argsort(outputs[0]).argsort()
    vaar_ranks = np.argsort(outputs[1]).argsort()
    
#            print(output_ranks)
#            print(vaar_ranks)
#            print("\n")
    rank_d.append(output_ranks[np.argmin(vaar_ranks)])
    
#            print("\n")
#            print("e sim\t{}".format("\t".join(map(lambda x: "{:.3f}".format(x),  outputs[0]))))
#            print("v sim\t{}".format("\t".join(map(lambda x: "{:.3f}".format(x),  outputs[1]))))
#            
#            print("\n")
#            print("e sim r\t{}".format("\t".join(map(lambda x: "{}".format(x),  
#                  np.argsort(outputs[0]).argsort()))))
#            print("v sim r\t{}".format("\t".join(map(lambda x: "{}".format(x),  
#                  np.argsort(outputs[1]).argsort()))))
#            print("\n")

print("\n")
print("Mean rank offset: {:.2f}".format(np.mean(rank_d)))
print("Rank matches    : {:.2f}%".format(
        (np.where(np.array(rank_d) == 0)[0].shape[0] * 100) / float(reps)))


end = time.time()
print("Duration: {:.2f}s".format(end - start))
#model.print_vars_distr()
