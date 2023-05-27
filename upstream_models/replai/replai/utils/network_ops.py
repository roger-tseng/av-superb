import torch.nn as nn


def remove_classification_head(model):
    if hasattr(model, "module"):
        model = model.module

    if hasattr(model, "fc"):
        model.fc = nn.Identity()
        return model

    if hasattr(model, "fc_audioset"):
        model.fc_audioset = nn.Identity()
        return model

    return model
