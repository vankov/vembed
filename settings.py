# -*- coding: utf-8 -*-

class Settings:
    N_SLOTS = 4
    MAX_ARITY = 2
    SEM_DIM = 10
    EMBDED_DIM = 50
    HIDDEN_UNITS_N = 100
    RECODE_HIDDEN_UNITS_N = 50
    LR = 0.0001
    BATCH_SIZE = 10
    TRAIN_STEPS_N = 5000
    TRAIN_EPOCHS_N = 10

    SIGMA = 0.5

    VARS_TOTAL_DIM = 0
    VARS_SEM_DIM = 0

    N_SYMBOLS = 1000
    BINARY_SYMBOLS_FREQ = 0.2
    UNARY_SYMBOLS_FREQ = 0.2
    ATOM_SYMBOLS_FREQ = 0.6

    EMBEDDINGS_FILENAME = "embeddings.txt"

Settings.VARS_TOTAL_DIM = Settings.N_SLOTS * (
    Settings.SEM_DIM + Settings.MAX_ARITY * (Settings.N_SLOTS))
Settings.VARS_SEM_DIM = Settings.N_SLOTS * Settings.SEM_DIM
