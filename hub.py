# Author: Yuan Tseng
# Hub for custom model classes

# Modified from S3PRL 
# (Authors: Leo Yang, Andy T. Liu and S3PRL team, https://github.com/s3prl/s3prl/blob/main/s3prl/hub.py)

from upstream_models.example.hubconf import *
# <<<<<<< HEAD
# =======
from upstream_models.replai.hubconf import *
from upstream_models.vhubert.hubconf import *
from upstream_models.avbert.hubconf import *
from upstream_models.mavil.hubconf import *
from upstream_models.hubert.hubconf import *
from upstream_models.fbank.hubconf import *
from upstream_models.hog.hubconf import *

# >>>>>>> origin/interface+avhubert+replai

def options(only_registered_ckpt: bool = False):
    all_options = []
    for name, value in globals().items():
        torch_hubconf_policy = not name.startswith("_") and callable(value)
        if torch_hubconf_policy and name != "options":
            if only_registered_ckpt and (
                name.endswith("_local")
                or name.endswith("_url")
                or name.endswith("_gdriveid")
                or name.endswith("_custom")
            ):
                continue
            all_options.append(name)

    return all_options
