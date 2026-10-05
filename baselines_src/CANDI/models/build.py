import torch

from Projects.AnomalyDetection.tvld.baselines_src.CANDI.models.timesnet.modeling_timesnet import TimesNet
from Projects.AnomalyDetection.tvld.baselines_src.CANDI.models.mlp.modeling_mlp import MLP


def build_model(cfg):
    model_name = cfg.MODEL.NAME

    model_mapping = {
        "TIMESNET": TimesNet,
        "MLP": MLP,
    }

    if model_name in model_mapping:
        model = model_mapping[model_name](cfg)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if torch.cuda.is_available():
        model = model.cuda()

    return model
