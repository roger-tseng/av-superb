from torch.utils.data._utils.collate import default_collate


def sequence_collate(batch):
    inputs = default_collate(batch)
    label = inputs[-2].flatten()
    index = inputs[-1].flatten()
    return inputs[:-2] + [label, index]


COLLATE_FN = {
    "VisualClassify": default_collate,
    "AudioClassify": default_collate,
    "MultimodalSequenceClassify": sequence_collate,
}
