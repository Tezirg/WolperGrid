import os
import json

class WolperGrid_Config():
    INITIAL_EPSILON = 1.0
    FINAL_EPSILON = 0.001
    DECAY_EPSILON = 256
    UNIFORM_EPSILON = False
    DISCOUNT_FACTOR = 0.99
    REPLAY_BUFFER_SIZE = 1024*48
    REPLAY_BUFFER_MIN = 1024
    LOG_FREQ = 128
    UPDATE_FREQ = 96
    UPDATE_TARGET_HARD_FREQ = -1
    UPDATE_TARGET_SOFT_TAU = 1e-4
    INPUT_BIAS = 0.0
    SAVE_FREQ = 1000
    K = 256
    BATCH_SIZE = 32
    LR_ACTOR = 1e-5
    LR_CRITIC = 1e-4
    GRADIENT_CLIP = False
    GRADIENT_INVERT = False
    VERBOSE = True
    ILLEGAL_GAME_OVER = False
    SIMULATE = -1
    SIMULATE_DO_NOTHING = False
    ACTION_SET_LINE = True
    ACTION_CHANGE_LINE = True
    ACTION_SET_BUS = True
    ACTION_CHANGE_BUS = True
    ACTION_REDISP = True

    @staticmethod
    def from_json(json_in_path):
        with open(json_in_path, 'r') as fp:
            conf_json = json.load(fp)
        
        for k,v in conf_json.items():
            if hasattr(WolperGrid_Config, k):
                setattr(WolperGrid_Config, k, v)

    @staticmethod
    def to_json(json_out_path):
        conf_json = {}
        for attr in dir(WolperGrid_Config):
            if attr.startswith('__') or callable(attr):
                continue
            conf_json[attr] = getattr(WolperGrid_Config, attr)

        with open(json_out_path, 'w+') as fp:
            json.dump(fp, conf_json, indent=2)
