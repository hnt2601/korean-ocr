from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

import paddle.fluid as fluid

import logging

logger = logging.getLogger(__name__)


def check_config_params(config, config_name, params):
    for param in params:
        if param not in config:
            err = "param %s didn't find in %s!" % (param, config_name)
            assert False, err
    return
