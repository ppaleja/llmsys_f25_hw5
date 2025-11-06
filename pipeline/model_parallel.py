import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, GPT2Model, GPT2PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)

from .model import GPT2LMHeadModelCustom, GPT2ModelCustom
from .partition import WithDevice, _retrieve_device
from .pipe import Pipe


class ExtractFirstItem(nn.Module):
    def __init__(self):
        super(ExtractFirstItem, self).__init__()

    def forward(self, x):
        return x[0]


class GPT2ModelParallel(GPT2ModelCustom):
    def __init__(self, config):
        super().__init__(config)

    def _prepare_pipeline_parallel(self, split_size=1):
        """
        Prepare the model for pipeline parallelism.

        Hint:
        1. Enable self.pipeline_parallel
        2. Construct an nn.Sequential module for the transformer layers (self.h).
        3. Use Pipe to parallelize the transformer layers.

        Please note that when implementing _prepare_pipeline_parallel, you would want to define the nn.Sequential module to extract useful values from the returned tuple. GPT2Block returns a tuple, not a tensor.
        You should construct nn.Sequential using GPT2Block modules. Notice that each block returns multiple values but you will only need the hidden states.
        """

        # BEGIN ASSIGN5_2_3
        # 1. Enable self.pipeline_parallel
        self.pipeline_parallel = True
        # 2. Construct an nn.Sequential module for the transformer layers (self.h).
        # Each GPT2Block returns a tuple, so we extract only the hidden states with ExtractFirstItem
        layers = []
        for block in self.h:
            dev = next(block.parameters()).device
            # Build layers with explicit device so ExtractFirstItem doesn’t default to CPU
            layers.extend([block, WithDevice(ExtractFirstItem(), dev)])
        pipe = Pipe(nn.Sequential(*layers), split_size=split_size)
        # END ASSIGN5_2_3
        self.h_pp = pipe
        # END ASSIGN5_2_3


class GPT2LMHeadModelParallel(GPT2LMHeadModelCustom):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config, GPT2ModelParallel(config))

    def _prepare_pipeline_parallel(self, split_size=1):
        self.parallelize()
        self.transformer._prepare_pipeline_parallel(split_size)

    def _finalize_pipeline_parallel(self):
        self.deparallelize()
        self.transformer.pipeline_parallel = False


if __name__ == "__main__":
    config = AutoConfig.from_pretrained("gpt2")
    # Use MPS if available, otherwise CPU
    device = "mps:0" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
    model = GPT2LMHeadModelParallel(config=config).to(device)
    model._prepare_pipeline_parallel()
