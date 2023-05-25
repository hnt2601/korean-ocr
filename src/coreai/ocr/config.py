cfg = {
    "checkpoint": "src/coreai/weights/str/lstm_model_best.pth.tar",
    "arch": "ResNet_ASTER",
    "len_voc": 0,
    "decoder_sdim": 512,
    "attDim": 512,
    "max_len": 15,
    "STN_ON": True,
    "tps_inputsize": [32, 64],
    "tps_outputsize": [32, 100],
    "eos": -1,
    "with_lstm": True,
    "n_group": 1,
    "network_id": -1,
    "path_configs_file": "",
    "nas_config_file": "",
    "num_control_points": 20,
    "tps_margins": [0.05, 0.05],
    "stn_activation": "none",
    "beam_width": 5,
}
