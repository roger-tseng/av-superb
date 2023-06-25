from .expert import UpstreamExpert as _UpstreamExpert

def mavil(refresh=False, *args, **kwargs):
    """
    The default model
        refresh (bool): whether to download ckpt/config again if existed
    """
    return _UpstreamExpert(*args, **kwargs)
