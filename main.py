# -*- coding: utf-8 -*-

from model import Model
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
#tf.compat.v1.enable_eager_execution()

model = Model()


model.train()

#print(model.summary())
#model.save_model_image("model.png")