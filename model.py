# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as K
import numpy as np
import sys
import pickle
from datetime import datetime

from scipy.special import factorial
from scipy.stats import spearmanr
from pathlib import Path

from settings import Settings
from data import VARSData

class Model:

    def train(self):
        logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = K.callbacks.TensorBoard(log_dir=logdir)

        train_gen = self._data.get_generator(
                        batch_size=Settings.BATCH_SIZE,
                        fixed_target=True)
        
        
        self._model.fit_generator(
                train_gen,
                steps_per_epoch=Settings.TRAIN_STEPS_N,
                epochs=Settings.TRAIN_EPOCHS_N,
                verbose=1,
#                validation_data=self._data.get_generator(
#                        batch_size=Settings.BATCH_SIZE,
#                        fixed_target=True),
#                validation_steps=100,
                callbacks=[tensorboard_callback],
                max_queue_size=1, 
                workers=1, 
                use_multiprocessing=False                
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
                dtype=np.int)
        state = np.full(shape=Settings.N_SLOTS, fill_value=-1, dtype=np.int)
        i = 0
        for positions in self.__construct_recode_mat_aux(0, state):
            recode_mat[i] = diag_mat[:][positions]
            i += 1
        return tf.convert_to_tensor(recode_mat, dtype=np.float32)

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
                recode_mat = tf.constant(pickle.load(file_h), dtype=np.float32)
            print("Done.")
            sys.stdout.flush()
        else:
            #create recoding matrix and serialize it to a file
            recode_mat = tf.constant(self.__construct_recode_mat(), dtype=tf.float32)
            with open(
                    "recode/recode_mat.{}.pickle".format(Settings.N_SLOTS),
                    "wb") as file_h:
                    pickle.dump(recode_mat.numpy(), file_h)

        return recode_mat

    def get_vars_comps(self, vars_tensor):
        vars_sem = tf.gather(
                tf.reshape(
                    vars_tensor,
                    [
                        Settings.BATCH_SIZE, 
                        Settings.N_SLOTS, 
                        Settings.SEM_DIM + Settings.MAX_ARITY * Settings.N_SLOTS
                    ]
                ),
                list(range(Settings.SEM_DIM)),
                axis=-1)

        vars_struct = tf.gather(
                tf.reshape(
                    vars_tensor,
                    [
                        Settings.BATCH_SIZE, 
                        Settings.N_SLOTS, 
                        Settings.SEM_DIM + Settings.MAX_ARITY * Settings.N_SLOTS                            
                    ]
                ),
                list(range(
                    Settings.SEM_DIM, 
                    Settings.SEM_DIM + Settings.MAX_ARITY * Settings.N_SLOTS)),
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

        #compute cosine similarity
        sem_cos = tf.norm(target_sem - base_sem, axis=-1)
        #-K.losses.cosine_similarity(target_sem, base_sem, axis=[2])
        #compute cosine similarity
        struct_cos = tf.norm(target_struct - base_struct, axis=-1)
        #-K.losses.cosine_similarity(target_struct, base_struct, axis=[2])
        
        similarities = tf.add(
                sem_cos * (1 - Settings.SIGMA), 
                struct_cos * Settings.SIGMA)
        
        #get maximum similarity
        max_similarities = tf.reduce_min(similarities, axis=-1)

        #get the indices of the recoding which maximize similarity
        best_recodings = tf.argmax(similarities, axis=-1)

        return max_similarities, best_recodings


    @tf.function
    def make_analogy(self, targets, bases):
                
        vaar_similarity, vaar_recoding = self._get_vaar(
                targets, bases)
        
        return vaar_similarity
    
    def _build_model(self):

        K.backend.set_floatx('float32')
        
        VARS_input = K.layers.Input(
                shape=(Settings.VARS_TOTAL_DIM,2,),
                batch_size=Settings.BATCH_SIZE,
                name='VARS_input')
        
        hidden_layer = K.layers.Dense(
                activation="relu",
#                kernel_regularizer=K.regularizers.l2(0.001),
                units=Settings.HIDDEN_UNITS_N,
                name='hidden_layer')        

        hidden_layer2 = K.layers.Dense(
                activation="relu",
#                kernel_regularizer=K.regularizers.l2(0.001),
                units=Settings.HIDDEN_UNITS_N,
                name='hidden_layer2')     
        
        embed_layer = K.layers.Dense(
                activation="linear",
                kernel_regularizer=K.regularizers.l2(0.001),
                units=Settings.EMBDED_DIM,
                name='embed_layer')


        vars1 = tf.squeeze(tf.gather(VARS_input, [0], axis=-1), axis=-1)
        vars2 = tf.squeeze(tf.gather(VARS_input, [1], axis=-1), axis=-1)

        hidden_state1 = hidden_layer2(hidden_layer(vars1))
        hidden_state2 = hidden_layer2(hidden_layer(vars2))

        embedding1 = embed_layer(hidden_state1)
        embedding2 = embed_layer(hidden_state2)


#        recode_hidden_layer = K.layers.Dense(
#                activation="linear",
#                units=Settings.RECODE_HIDDEN_UNITS_N,
#                name='recode_hidden_layer')(
#                        K.layers.concatenate(
#                                [embedding1, embedding2], axis=-1))

        euclidian_dist = tf.norm(embedding1 - embedding2, axis=-1)
#        print(euclidian_dist)
        
        output_similarity = K.layers.Lambda(
                    lambda x: x, 
                    name="e_sim"
                )(euclidian_dist)
#        K.layers.dot(
#                [embedding1, embedding2],
#                axes=-1,
#                normalize=False,
#                name="e_sim")

#        output_recode = K.layers.Dense(
#                units = self._n_states,
#                activation = "softmax",
#                name="recode_output")(recode_hidden_layer)

        vaar_similarity, vaar_recoding = self._get_vaar(vars1, vars2)

        
        sim_tensor = tf.stack(
                [
                        output_similarity, 
                        vaar_similarity
                ])
    
        sim_output = K.layers.Lambda(lambda x: x, name="sim")(sim_tensor)

#        similarity_corel = K.layers.Lambda(
#                lambda x : x,
#                name="corel")(tfp.stats.correlation(
#                                output_similarity,
#                                tf.expand_dims(vaar_similarity, 1),
#                                sample_axis=0,
#                                event_axis=None))
#        
#        similarity_mse = K.layers.Lambda(
#                lambda x : x,
#                name="mse")(K.losses.mse(
#                                output_similarity,
#                                tf.expand_dims(vaar_similarity, 1)))
#        
#        recode_match = K.layers.Lambda(
#                lambda x : x,
#                name="recode")(K.losses.mse(
#                                output_recode,
#                                tf.one_hot(vaar_recoding, self._n_states)))
        self._model = K.Model(
                inputs=[VARS_input],
                outputs=[sim_output],
                )

        def pearson_corel(y_true, y_pred):
            return tfp.stats.correlation(
                    y_pred[0,:],
                    y_pred[1,:],
                    sample_axis=0,
                    event_axis=None)

        def spearman_corel(y_true, y_pred):
            return tf.py_function(
                        spearmanr, 
                        [
                            tf.cast(y_pred[0, :], tf.float32), 
                            tf.cast(y_pred[1, :], tf.float32)
                        ], 
                        Tout = tf.float32)
            
        def std_vaars(y_true, y_pred):
            return tf.math.reduce_min(tf.math.l2_normalize(y_pred[1]))

        def std_emd(y_true, y_pred):
            return tf.math.reduce_max(tf.math.l2_normalize(y_pred[1]))
        
        def rank_match(y_true, y_pred):
            
            e_ranks = tf.cast(tf.argsort(tf.argsort(y_pred[0,:])), tf.float32)
            v_ranks = tf.cast(tf.argsort(tf.argsort(y_pred[1,:])), tf.float32)
            ranks_diff = tf.abs(e_ranks-v_ranks)
            
            return tf.reduce_mean(ranks_diff) / Settings.BATCH_SIZE
#                        tf.math.less(ranks_diff, 2), dtype=tf.float32)
#                        ) / e_ranks.shape[0]
                
        #tf.reduce_mean(tf.cast(ranks_diff, dtype=tf.float32))
#                    tf.cast(tf.math.less(ranks_diff, 5), dtype=tf.float32))
            #tf.cast(tf.constant(list(range(0, 100))), tf.float32)
            #tf.cast(tf.argsort(y_pred[1,:]), tf.float32)
            
#            return tf.reduce_mean(-K.losses.cosine_similarity(e_ranks, v_ranks))
#        tf.reduce_mean(
#                    tf.cast(tf.abs(e_ranks - v_ranks), dtype=tf.float32))

        def top_1(y_true, y_pred):
            return tf.cast(
                    tf.equal(
                        tf.argmin(y_pred[0, :]),
                        tf.argmin(y_pred[1, :])),
                    tf.float32)
#        K.metrics.top_k_categorical_accuracy(
#                        tf.expand_dims(
#                            tf.one_hot(
#                                tf.argmax(y_pred[1,:]), 
#                                y_pred[1,:].shape[0],
#                                dtype=tf.float32), axis=0),
#                        tf.expand_dims(y_pred[0,:], axis=0), 
#                        k=1)
        
        def top1_d(_, y_pred):
            s_out = y_pred[0, :]
            v_out = y_pred[1, :]
            s_r = tf.argsort(tf.argsort(s_out))
            v_r = tf.argsort(tf.argsort(v_out))

            v_min_arg = tf.argmin(v_r)
            return tf.abs(
                        tf.gather(v_r, v_min_arg) 
                        - tf.gather(s_r, v_min_arg)) / v_out.shape[0]
        
        def loss_f(y_true, y_pred):
            
            s_out = y_pred[0, :]
            v_out = y_pred[1, :]
            s_r = tf.argsort(tf.argsort(s_out))
            v_r = tf.argsort(tf.argsort(v_out))
            
            r_diff = tf.cast(s_r - v_r, tf.float32)
            
            return K.losses.mse(
                    s_out,         
                     tf.cast(tf.less(r_diff, 0), tf.float32) + -1 * tf.cast(tf.greater(r_diff, 0), tf.float32)
                )
#        1\
#                K.losses.mse(
#                        tf.nn.softmax(y_pred[0,:]), 
#                        tf.nn.softmax(y_pred[1,:]))
#        K.losses.mse(
#                        y_pred[1,:],
#                        y_pred[0,:]
#                    )
#        tf.nn.softmax_cross_entropy_with_logits(y_pred[1,:], y_pred[0,:])
        #K.losses.mse(y_pred[0,:], y_pred[1,:])
#        -tfp.stats.correlation(
#                    y_pred[0,:],
#                    y_pred[1,:],
#                    sample_axis=0,
#                    event_axis=None)
#            
        self._model.compile(
                loss={
                    #"mse": lambda y_true, y_pred: 0.0,#tf.reduce_mean(y_pred),
                    "sim": lambda y_true, y_pred:-pearson_corel(y_true, y_pred),#lambda y_true, y_pred: K.losses.mse(y_pred[0,:], y_pred[1,:]),
                },
                optimizer=K.optimizers.RMSprop(lr = Settings.LR),
                metrics={"sim": [spearman_corel, top_1, top1_d]}
            
                )

    def save(self):
        self._model.save_weights("last.model.h5")

    def load(self, filename="last.model.h5"):
        self._model.load_weights(filename)
        
    def print_vars_distr(self):
        data = next(self._data.get_generator(
                        batch_size=100,
                        fixed_target=True))
        
        
        print(np.mean(data[0][:,:,1]))
        print(np.std(data[0][:,:,1]))
        print(np.min(data[0][:,:,1]))
        print(np.max(data[0][:,:,1]))
        
        exit(0)
        
    def test(self, data_gen):
        
        while(True):
            yield self._model.predict(
                    data_gen, 
                    steps=1, 
                    max_queue_size=10, 
                    workers=10, 
                    use_multiprocessing=False)
        
    def save_model_image(self, filename):
        K.utils.plot_model(self._model, to_file=filename)

    def __init__(self):

        self._n_states = int(factorial(Settings.N_SLOTS))
        self._recode_mat = self._get_recode_matrix()
        self._build_model()
        self._data = VARSData()

