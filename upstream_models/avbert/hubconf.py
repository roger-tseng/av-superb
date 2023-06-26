import os

from util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert

def google_large_file_link(file_id):
    return f"https://drive.google.com/u/0/uc?id={file_id}&export=download&confirm=t&uuid=b73dc818-c9b0-49e8-a488-13b829fdbb7e&at=ANzk5s4TfW9_erYs4M9PH7KYuKzy:1681055195563"

def avbert_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def avbert_url(ckpt, refresh=False, *args, **kwargs):
    """
    The model from URL
        ckpt (str): URL
    """
    return avbert_local(_urls_to_filepaths(ckpt, refresh=refresh, agent="gdown"), *args, **kwargs)

def avbert_time(refresh=False, *args, **kwargs):
    """
    Interleave features of three video crops along time axis
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = google_large_file_link("1T55DuJq0OYSdXVnDlOrgPSTwffVKsiFY")
    kwargs["feature_concat_axis"] = 'time'
    return avbert_url(refresh=refresh, *args, **kwargs)

def avbert_hidden(refresh=False, *args, **kwargs):
    """
    Concat features of three video crops along hidden dim axis
        refresh (bool): whether to download ckpt/config again if existed
    """
    kwargs["ckpt"] = google_large_file_link("1T55DuJq0OYSdXVnDlOrgPSTwffVKsiFY")
    kwargs["feature_concat_axis"] = 'hidden'
    return avbert_url(refresh=refresh, *args, **kwargs)

def avbert(refresh=False, *args, **kwargs):
    """
    Interleave features along time axis by default
    """
    return avbert_time(refresh=refresh, *args, **kwargs)
