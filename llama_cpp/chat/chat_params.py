import sys, os, datetime
from time import time
from os import cpu_count, path
import llama_cpp
import os
import argparse
import re
from dataclasses import dataclass, field
from typing import List
import ctypes
import sys
from time import time
from os import cpu_count, path


@dataclass
class GptParams:
    seed: int = -1
    n_threads: int = min(4, os.cpu_count() or 1)
    n_predict: int = 128
    n_parts: int = -1
    n_ctx: int = 512
    n_batch: int = 8
    n_keep: int = 0

    ignore_eos: bool = False
    logit_bias: dict[int, float] = field(default_factory=dict)
    top_k: int = 40
    top_p: float = 0.95
    tfs_z: float = 1.00
    typical_p: float = 1.00
    temp: float = 0.80
    repeat_penalty: float = 1.10
    repeat_last_n: int = 64
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    mirostat: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1

    model: str = "./models/llama-7B/ggml-model.bin"
    prompt: str = ""
    path_session: str = ""
    input_prefix: str = " "
    input_suffix: str = ""
    antiprompt: List[str] = field(default_factory=list)

    lora_adapter: str = ""
    lora_base: str = ""

    memory_f16: bool = True
    random_prompt: bool = False
    use_color: bool = False
    interactive: bool = False

    embedding: bool = False
    interactive_start: bool = False

    instruct: bool = False
    penalize_nl: bool = True
    perplexity: bool = False
    use_mmap: bool = True
    use_mlock: bool = False
    mem_test: bool = False
    verbose_prompt: bool = False

    file: str = None

    # If chat ended prematurely, append this to the conversation to fix it.
    # Set to "\nUser:" etc.
    # This is an alternative to input_prefix which always adds it, so it potentially duplicates "User:""
    fix_prefix: str = ""
    input_echo: bool = (True,)

    # Default instructions for Alpaca
    # switch to "Human" and "Assistant" for Vicuna.
    # TODO: TBD how they are gonna handle this upstream
    instruct_inp_prefix: str = "\n\n### Instruction:\n\n"
    instruct_inp_suffix: str = "\n\n### Response:\n\n"
