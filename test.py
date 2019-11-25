# -*- coding: utf-8 -*-
import tensorflow as tf
import time

from model import Model


tf.compat.v1.disable_eager_execution()
#tf.compat.v1.enable_eager_execution()


model = Model()

model.load()

start = time.time()
model.test()
end = time.time()
print("Duration: {:.2f}s".format(end - start))
#model.print_vars_distr()
