# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import sys
import pickle
from datetime import datetime

from scipy.special import factorial
from pathlib import Path

from settings import Settings

#class VAARSimilarityLoss(K.losses.Loss):
#    
#    def __init__(self, output_similarity, reduction=tf.keras.losses.Reduction.AUTO, name='VAARSimilarityLoss'):
#        super().__init__(reduction=reduction, name=name)
#        self.output_similarity = output_similarity
#
#    def sdr_loss(self, sig_true, sig_pred):
#        return (-tf.reduce_mean(sig_true * sig_pred) /
#                (tf.norm(tensor=sig_pred) * tf.norm(tensor=sig_true)))
#
#    def call(self, y_true, y_pred):
#        noise_true = self.noisy_signal - y_true
#        noise_pred = self.noisy_signal - y_pred
#        alpha = (tf.reduce_mean(tf.square(y_true)) /
#                 tf.reduce_mean(tf.square(y_true) + tf.square(self.noisy_signal - y_pred)))
#        return 
#    
#    alpha * self.sdr_loss(y_true, y_pred) + (1 - alpha) * self.sdr_loss(noise_true, noise_pred)
    
class Model:

        
    def _vars_input_generator(self):
        
        while(True):
            vars_input = np.random.uniform(
                low=0,
                high=1,
                size=(
                    Settings.BATCH_SIZE,
                    Settings.VARS_TOTAL_DIM,
                    2))
                
            sim_targets = np.zeros(
                shape=(
                    Settings.BATCH_SIZE,
                    1))
            
            recode_targets = np.zeros(
                shape=(
                    Settings.BATCH_SIZE,
                    self._n_states))            
            
            yield vars_input, [sim_targets, recode_targets]
            
    def train(self):
        logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = K.callbacks.TensorBoard(log_dir=logdir)

        self._model.fit_generator(
                self._vars_input_generator(), 
                steps_per_epoch=Settings.TRAIN_STEPS_N, 
                epochs=Settings.TRAIN_EPOCHS_N, 
                verbose = 1,
                callbacks=[tensorboard_callback]
            )
        
    def summary(self):
        return self._model.summary()
    
    def __construct_recode_mat_aux(self, slot_no, state):
        """
            Auxiliary function
        """
        if slot_no >= Settings.N_SLOTS:
            yield state
    
        for slot_i in [x for x in range(Settings.N_SLOTS) if not x in state]:
            state_copy = np.copy(state)
            state_copy[slot_no] = slot_i
    
            for state_t in self.__construct_recode_mat_aux(
                    slot_no + 1, 
                    state_copy):
                yield state_t
                                
    def __construct_recode_mat(self):
        """
            Construct a new recoding matrix
        """
        diag_mat = np.diagflat(np.ones(shape=(Settings.N_SLOTS)))
        recode_mat = np.zeros(
                shape=(self._n_states, Settings.N_SLOTS, Settings.N_SLOTS),
                dtype=np.float32)
        state = np.full(shape=Settings.N_SLOTS, fill_value=-1, dtype=np.int)
        i = 0
        for positions in self.__construct_recode_mat_aux(0, state):
            recode_mat[i] = diag_mat[:][positions]
            i += 1
        return tf.convert_to_tensor(recode_mat)

    def _get_recode_matrix(self):
        """
            Loads the recoding matrix from "./recode". If not found, creates
            a new one
        """
                
        if Path("recode/recode_mat.{}.pickle".format(Settings.N_SLOTS)).is_file():
            #a recoding matrix with given parameters is already 
            # created and serialized, load it
            print("Loading recoding matix...", end="")
            sys.stdout.flush()
            with open(
                    "recode/recode_mat.{}.pickle".format(Settings.N_SLOTS), 
                    "rb") as file_h:                            
                recode_mat = tf.constant(pickle.load(file_h))
            print("Done.")
            sys.stdout.flush()
        else:
            #create recoding matrix and serialize it to a file
            recode_mat = tf.constant(self.__construct_recode_mat())
            with open(
                    "recode/recode_mat.{}.pickle".format(Settings.N_SLOTS), 
                    "wb") as file_h:
                    pickle.dump(recode_mat.numpy(), file_h)   
                    
        return recode_mat
    
    def get_vars_comps(self, vars_tensor):
        vars_sem = tf.gather(
                tf.reshape(
                    vars_tensor,
                    [Settings.BATCH_SIZE, Settings.VARS_TOTAL_DIM]
                ),
                list(range(Settings.VARS_SEM_DIM)),
                axis=-1)
                
        vars_struct = tf.gather(
                tf.reshape(
                    vars_tensor,
                    [Settings.BATCH_SIZE, Settings.VARS_TOTAL_DIM]
                ),
                list(range(Settings.VARS_SEM_DIM, Settings.VARS_TOTAL_DIM)),
                axis=-1)            
                
        return vars_sem, vars_struct
    
    @tf.function
    def _get_vaar(self, target_vars, base_vars):
        """            
            Build TF graph of VAAR (the Vector Approach to Analogical Reasoning)
        """

        recode_sem_target_func = \
            lambda vars_sem: \
                tf.reshape(
                    tf.matmul(
                        tf.reshape(
                            self._recode_mat,
                            [
                                self._n_states * Settings.N_SLOTS,
                                Settings.N_SLOTS
                            ]),
                        tf.reshape(
                                vars_sem,
                                [Settings.N_SLOTS, Settings.SEM_DIM])),                                
                    [self._n_states, Settings.N_SLOTS * Settings.SEM_DIM])

        recode_struct_target_func = \
            lambda vars_struct: \
                tf.reshape( 
                    tf.transpose(
                        tf.reshape(
                            tf.concat(
                                [                            
                                    tf.matmul(
                                        tf.reshape(
                                            tf.matmul(
                                                tf.reshape(
                                                    self._recode_mat, 
                                                    [
                                                        (self._n_states 
                                                         * Settings.N_SLOTS), 
                                                        Settings.N_SLOTS
                                                    ]), 
                                                tf.gather(
                                                    tf.reshape(
                                                        vars_struct,
                                                        [
                                                            Settings.MAX_ARITY,
                                                            Settings.N_SLOTS,
                                                            Settings.N_SLOTS
                                                        ]),                                                    
                                                    [a_i],
                                                    axis=0)
                                            ), 
                                            [
                                                self._n_states, 
                                                Settings.N_SLOTS, 
                                                Settings.N_SLOTS
                                            ]),
                                        tf.transpose(self._recode_mat, [0, 2, 1]),
                                    )
                                    for a_i in range(Settings.MAX_ARITY)
                                ], 0
                            ),
                            [
                                Settings.MAX_ARITY, 
                                self._n_states, 
                                tf.square(Settings.N_SLOTS)
                            ]), 
                        [1, 0, 2]),
                    [
                        self._n_states, 
                        Settings.MAX_ARITY * tf.square(Settings.N_SLOTS)
                    ])

        
        target_sem, target_struct = self.get_vars_comps(target_vars)
        base_sem, base_struct = self.get_vars_comps(base_vars)
        
        #generate all possible states of the semantics of the target                
        target_sem = tf.map_fn(recode_sem_target_func, target_sem)
        
        #generate all possible states of the structure of the target
        target_struct = tf.map_fn(recode_struct_target_func, target_struct)
        
        #reshapoe bases
        base_sem = tf.reshape(
            tf.tile(
                    tf.reshape(
                            base_sem, 
                            [
                                Settings.BATCH_SIZE,
                                Settings.N_SLOTS * Settings.SEM_DIM
                            ]),
                    [1, self._n_states]),
            [Settings.BATCH_SIZE, self._n_states, Settings.VARS_SEM_DIM])
        
        base_struct = tf.reshape(
            tf.tile(
                    tf.reshape(
                        base_struct, 
                        [
                            Settings.BATCH_SIZE, 
                            Settings.MAX_ARITY * tf.square(Settings.N_SLOTS)
                        ]),
                    [1, self._n_states]),
            [
                Settings.BATCH_SIZE, 
                self._n_states, 
                Settings.MAX_ARITY * tf.square(Settings.N_SLOTS)
            ])
        
        #compute semantics denominator for cosine similarity
        denom_sem = tf.multiply( 
            tf.sqrt(
                tf.reduce_sum(
                    tf.multiply(target_sem, target_sem), 
                    axis=[2])),                
            tf.sqrt(
                tf.reduce_sum(
                    tf.multiply(base_sem, base_sem), 
                    axis=[2])))      
            
        #compute numerator
        num_sem = tf.reduce_sum(tf.multiply(target_sem, base_sem), axis=[2])        
        #compute cosine similarity
        sem_cos = tf.add(tf.multiply(tf.divide(num_sem, denom_sem), 0.5), 0.5) 
                
        #compute structure denominator for cosine similarity
        denom_struct = tf.reduce_sum(target_struct, axis=[2])
        #compute numerator
        num_struct = tf.reduce_sum(
                tf.multiply(target_struct, base_struct), axis=[2])
        #compute cosine similarity
        struct_cos = tf.divide(num_struct, denom_struct)
        
        similarities = tf.add(
                tf.multiply(sem_cos, 1 - Settings.SIGMA), 
                tf.multiply(struct_cos, Settings.SIGMA))

        self._sem_cos = sem_cos
        self._struct_cos = struct_cos
        
        #get maximum similarity
        max_similarities = tf.reduce_max(similarities, axis=-1)
        
        #get the indices of the recoding which maximize similarity
        best_recodings = tf.argmax(similarities, axis=-1)

        return max_similarities, best_recodings
    

    def _build_model(self):

        VARS_input = K.layers.Input(
                shape=(Settings.VARS_TOTAL_DIM,2,),
                batch_size=Settings.BATCH_SIZE,
                name='VARS_input')
                
        hidden_layer = K.layers.Dense(
                activation="relu",        
                units=Settings.HIDDEN_UNITS_N,
                name='hidden_layer')
        
        embed_layer = K.layers.Dense(
                activation="tanh",        
                units=Settings.EMBDED_DIM,
                name='embed_layer')
        
        
        vars1 = tf.squeeze(tf.gather(VARS_input, [0], axis=-1), axis=-1)
        vars2 = tf.squeeze(tf.gather(VARS_input, [1], axis=-1), axis=-1)
        
        hidden_state1 = hidden_layer(vars1)        
        hidden_state2 = hidden_layer(vars2)
        
        embedding1 = embed_layer(hidden_state1)
        embedding2 = embed_layer(hidden_state2)

        recode_hidden_layer = K.layers.Dense(
                activation="tanh",        
                units=Settings.RECODE_HIDDEN_UNITS_N,
                name='recode_hidden_layer')(
                        K.layers.concatenate(
                                [embedding1, embedding2], axis=-1))

        output_similarity = K.layers.dot(
                [embedding1, embedding2], 
                axes=-1,
                normalize=True,
                name="similarity_output")
                
        output_recode = K.layers.Dense(
                units = self._n_states,
                activation = "softmax",
                name="recode_output")(recode_hidden_layer)

        vaar_similarity, vaar_recoding = self._get_vaar(vars1, vars2)
     
        similarity_match = K.layers.Lambda(
                lambda x : x, 
                name="sim")(K.losses.mse(
                                output_similarity,
                                tf.expand_dims(vaar_similarity, 1)))
        
        recode_match = K.layers.Lambda(
                lambda x : x, 
                name="recode")(K.losses.categorical_crossentropy(
                                output_recode,
                                tf.one_hot(vaar_recoding, self._n_states)))
        
        self._model = K.Model(
                inputs=[VARS_input],
                outputs=[similarity_match, recode_match]
                )
        
        self._model.compile(
                loss=lambda y_true, y_pred: tf.reduce_mean(y_pred), 
                optimizer=K.optimizers.RMSprop(lr = Settings.LR),
                metrics={"recode": "acc"}
                )
                    
    def save(self):
        pass

    def save_model_image(self, filename):
        K.utils.plot_model(self._model, to_file=filename)
        
    def __init__(self):    
        
        self._n_states = int(factorial(Settings.N_SLOTS))            
        self._recode_mat = self._get_recode_matrix()
        self._build_model()
        