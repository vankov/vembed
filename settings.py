# -*- coding: utf-8 -*-

class Settings:
    N_SLOTS = 6
    MAX_ARITY = 2
    SEM_DIM = 50
    EMBDED_DIM = 128 * 4
    HIDDEN_UNITS_N = 1024 * 2
    RECODE_HIDDEN_UNITS_N = 50
    LR = 0.0001
    BATCH_SIZE = 2000
    TRAIN_STEPS_N = 1000
    TRAIN_EPOCHS_N = 10

    SIGMA = 0.7

    VARS_TOTAL_DIM = 0
    VARS_SEM_DIM = 0

    N_SYMBOLS = 100000
    BINARY_SYMBOLS_FREQ = 0.2
    UNARY_SYMBOLS_FREQ = 0.2
    ATOM_SYMBOLS_FREQ = 0.6

    EMBEDDINGS_FILENAME = "embeddings.npy"

Settings.VARS_TOTAL_DIM = Settings.N_SLOTS * (
    Settings.SEM_DIM + Settings.MAX_ARITY * (Settings.N_SLOTS))
Settings.VARS_SEM_DIM = Settings.N_SLOTS * Settings.SEM_DIM
