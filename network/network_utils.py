from network.model_ct_v29 import CTV29

def build_model(cfg):
    net = eval(cfg.model_name)(cfg)
    return net
