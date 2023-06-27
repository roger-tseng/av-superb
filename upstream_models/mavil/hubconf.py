import os

from utils.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


def google_large_file_link(file_id):
    return f"https://drive.google.com/u/0/uc?id={file_id}&export=download&confirm=t&uuid=b73dc818-c9b0-49e8-a488-13b829fdbb7e&at=ANzk5s4TfW9_erYs4M9PH7KYuKzy:1681055195563"


def mavil_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def mavil_url(ckpt, refresh=False, *args, **kwargs):
    """
    The model from URL
        ckpt (str): URL
    """
    return mavil_local(
        _urls_to_filepaths(ckpt, refresh=refresh, agent="gdown"), *args, **kwargs
    )


def mavil_base(refresh=False, *args, **kwargs):
    """
    The pretrained model
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = google_large_file_link("160pJQDQGlNwIb5xWlSc73A713I4uXCtg")
    return mavil_url(refresh=refresh, *args, **kwargs)
