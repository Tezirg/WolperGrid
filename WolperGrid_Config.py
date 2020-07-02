import os
import json

class WolperGrid_Config():
    INITIAL_EPSILON = 1.0
    FINAL_EPSILON = 0.001
    DECAY_EPSILON = 256
    UNIFORM_EPSILON = False
    DISCOUNT_FACTOR = 0.99
    REPLAY_BUFFER_SIZE = 1024*48
    LOG_FREQ = 100
    UPDATE_FREQ = 96
    UPDATE_TARGET_HARD_FREQ = -1
    UPDATE_TARGET_SOFT_TAU = 1e-3
    INPUT_BIAS = 0.0
    SAVE_FREQ = 100
    K = 256
    BATCH_SIZE = 32
    LR = 1e-4
    GRADIENT_CLIP = False
    VERBOSE = True
    ILLEGAL_GAME_OVER = False
    SIMULATE = -1
    SIMULATE_DO_NOTHING = False

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
