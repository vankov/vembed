# -*- coding: utf-8 -*-
import tensorflow as tf

from model import Model


tf.compat.v1.disable_eager_execution()
#tf.compat.v1.enable_eager_execution()


model = Model()

model.load()
model.test()