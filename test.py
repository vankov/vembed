# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

import time

from model import Model
from data import VARSData
from settings import Settings

#tf.compat.v1.disable_eager_execution()
tf.compat.v1.enable_eager_execution()
model = None

def run_rest(test):
    global model
#    print(test.shape)
    
    batch_size = test.shape[0]

    Settings.BATCH_SIZE = batch_size

    if model is None:
        model = Model()
    
    def gen_data():
        while(True):
            yield test
    
    data = gen_data()
    
    d = next(data)
    d = np.float32(d)
#    print(d.shape)
#    print(d.reshape(
#            5,
#            Settings.N_SLOTS,
#            (
#                    Settings.SEM_DIM
#                    + Settings.MAX_ARITY * (
#                            Settings.N_SLOTS
#                        )
#                ),
#            2, 
#            )[0,:,:3, 0]
#        )
#              
#    print(d.reshape(
#            5,
#            Settings.N_SLOTS,
#            (
#                    Settings.SEM_DIM
#                    + Settings.MAX_ARITY * (
#                            Settings.N_SLOTS
#                        )
#                ),
#            2,
#            )[0,:,:3,1]
#        )
#    t = np.copy(d[1:2,:,0]).reshape(
#            Settings.BATCH_SIZE,
#            Settings.N_SLOTS,
#            Settings.SEM_DIM + Settings.MAX_ARITY * Settings.N_SLOTS
#        )        
#    t2 = np.copy(d[1:2,:,1]).reshape(
#            Settings.BATCH_SIZE,
#            Settings.N_SLOTS,
#            Settings.SEM_DIM + Settings.MAX_ARITY * Settings.N_SLOTS
#        )
#    t = np.float32(np.random.uniform(    
#            size=(3, 10)
#        ))
#    t[:, :] = 0
#    t[0,:4] = [1, 2, 3, 4]
#    t[1,:4] = [5, 6, 7, 8]
#    t[2,:4] = [9, 10, 11, 12]
#    
#    t[0,4:] = [1, 0, 1, 0, 1, 0]
#    t[1,4:] = [0, 2, 0, 2, 0, 2]
#    t[2,4:] = [1, 0, 0, 0, 0, 1]
    
#    t2 = np.copy(t)
#    r = t2[0,:,:]
#    np.random.shuffle(r)
#    t2[0,:,:] = r
#    np.random.shuffle(t2)         
#    t = np.expand_dims(t, axis=0)
#    t2 = np.expand_dims(t2, axis=0)
#    for i in range(Settings.N_SLOTS):
#        t2[:,i,Settings.SEM_DIM:] = i
#    t2[:,:,Settings.SEM_DIM:] = 1
#    print(t.shape)
#    print(t2.shape)
#    print(t2[:,:,:3])
#    print(t2[:,:,Settings.SEM_DIM:])
#    
#    t = np.zeros(shape=(
#                2,
#                Settings.N_SLOTS,
#                Settings.SEM_DIM
#                + Settings.MAX_ARITY * Settings.N_SLOTS
#            ))
#    t2 = np.copy(t)
#    
#    for i in range(Settings.N_SLOTS):
#        t[:,i,0:Settings.SEM_DIM] = np.array([1, 2, 3, 4, 5]) + i * 10
#    
#    t[:,0,Settings.SEM_DIM + 1] = 1
#    t[:,0,Settings.SEM_DIM + Settings.N_SLOTS + 2] = 1
#
#    t2[:,1,Settings.SEM_DIM] = 1
#    t2[:,1,Settings.SEM_DIM + Settings.N_SLOTS + 2] = 1
##
##    
#    t = np.float32(t)
#    t2 = np.float32(t2)
    ma = model.make_analogy(d[:,:,0], d[:,:,1])
    
    print(ma)
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


test1 = np.load("test.1.npy")
test2 = np.load("test.2.npy")

            
run_rest(test1)
run_rest(test2)

#exit(0)
#
##model = Model()
#
##model.load()
#
#start = time.time()
#rank_d = []
#
#reps = 100
#
#varsData = VARSData()
#data_gen = varsData.get_generator(
#        batch_size=Settings.BATCH_SIZE,
#        fixed_target=True)
#
#for _ in range(reps):
#    outputs = next(model.test(data_gen))
#    
#    
#    output_ranks = np.argsort(outputs[0]).argsort()
#    vaar_ranks = np.argsort(outputs[1]).argsort()
#    
##            print(output_ranks)
##            print(vaar_ranks)
##            print("\n")
#    rank_d.append(output_ranks[np.argmin(vaar_ranks)])
#    
##            print("\n")
##            print("e sim\t{}".format("\t".join(map(lambda x: "{:.3f}".format(x),  outputs[0]))))
##            print("v sim\t{}".format("\t".join(map(lambda x: "{:.3f}".format(x),  outputs[1]))))
##            
##            print("\n")
##            print("e sim r\t{}".format("\t".join(map(lambda x: "{}".format(x),  
##                  np.argsort(outputs[0]).argsort()))))
##            print("v sim r\t{}".format("\t".join(map(lambda x: "{}".format(x),  
##                  np.argsort(outputs[1]).argsort()))))
##            print("\n")
#
#print("\n")
#print("Mean rank offset: {:.2f}".format(np.mean(rank_d)))
#print("Rank matches    : {:.2f}%".format(
#        (np.where(np.array(rank_d) == 0)[0].shape[0] * 100) / float(reps)))
#
#
#end = time.time()
#print("Duration: {:.2f}s".format(end - start))
##model.print_vars_distr()
