import sys
import os
import datetime
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

from llama_cpp.chat.chat_api import LLaMAInteract
from .chat_params import GptParams


def gpt_params_parse(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=-1,
        help="RNG seed (use random seed for <= 0)",
        dest="seed",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="number of threads to use during computation",
        dest="n_threads",
    )
    parser.add_argument(
        "-n",
        "--n_predict",
        type=int,
        default=128,
        help="number of tokens to predict (-1 = infinity)",
        dest="n_predict",
    )
    parser.add_argument(
        "--n_parts", type=int, default=-1, help="number of model parts", dest="n_parts"
    )
    parser.add_argument(
        "-c",
        "--ctx_size",
        type=int,
        default=512,
        help="size of the prompt context",
        dest="n_ctx",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=8,
        help="batch size for prompt processing",
        dest="n_batch",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=0,
        help="number of tokens to keep from the initial prompt",
        dest="n_keep",
    )

    parser.add_argument(
        "-l",
        "--logit-bias",
        type=str,
        action="append",
        help="--logit-bias TOKEN_ID(+/-)BIAS",
        dest="logit_bias_str",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="ignore end of stream token and continue generating",
        dest="ignore_eos",
    )
    parser.add_argument(
        "--top_k", type=int, default=40, help="top-k sampling", dest="top_k"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95, help="top-p samplin", dest="top_p"
    )
    parser.add_argument(
        "--tfs",
        type=float,
        default=1.0,
        help="tail free sampling, parameter z (1.0 = disabled)",
        dest="tfs_z",
    )
    parser.add_argument(
        "--temp", type=float, default=0.80, help="temperature", dest="temp"
    )
    parser.add_argument(
        "--repeat_penalty",
        type=float,
        default=1.10,
        help="penalize repeat sequence of tokens",
        dest="repeat_penalty",
    )
    parser.add_argument(
        "--repeat_last_n",
        type=int,
        default=64,
        help="last n tokens to consider for penalize ",
        dest="repeat_last_n",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=0.0,
        help="repeat alpha frequency penalty (0.0 = disabled)",
        dest="tfs_z",
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0.0,
        help="repeat alpha presence penalty (0.0 = disabled)",
        dest="presence_penalty",
    )
    parser.add_argument(
        "--mirostat",
        type=float,
        default=1.0,
        help="use Mirostat sampling.",
        dest="mirostat",
    )
    parser.add_argument(
        "--mirostat_ent",
        type=float,
        default=5.0,
        help="Mirostat target entropy, parameter tau represents the average surprise value",
        dest="mirostat_tau",
    )
    parser.add_argument(
        "--mirostat_lr",
        type=float,
        default=0.1,
        help="Mirostat learning rate, parameter eta",
        dest="mirostat_eta",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="./models/llama-7B/ggml-model.bin",
        help="model path",
        dest="model",
    )
    parser.add_argument(
        "-p", "--prompt", type=str, default="", help="initial prompt", dest="prompt"
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=None,
        help="file containing initial prompt to load",
        dest="file",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="file to cache model state in (may be large!)",
        dest="path_session",
    )
    parser.add_argument(
        "--in-prefix",
        type=str,
        default="",
        help="string to prefix user inputs with",
        dest="input_prefix",
    )
    parser.add_argument(
        "--in-suffix", type=str, default="", help="append to input", dest="input_suffix"
    )
    parser.add_argument(
        "-r",
        "--reverse-prompt",
        type=str,
        action="append",
        help="poll user input upon seeing PROMPT (can be\nspecified more than once for multiple prompts).",
        dest="antiprompt",
    )

    parser.add_argument(
        "--lora",
        type=str,
        default="",
        help="apply LoRA adapter (implies --no-mmap)",
        dest="lora_adapter",
    )
    parser.add_argument(
        "--lora-base",
        type=str,
        default="",
        help="optional model to use as a base for the layers modified by the LoRA adapter",
        dest="lora_base",
    )

    parser.add_argument(
        "--memory_f32",
        action="store_false",
        help="use f32 instead of f16 for memory key+value",
        dest="memory_f16",
    )
    parser.add_argument(
        "--random-prompt",
        action="store_true",
        help="start with a randomized prompt.",
        dest="random_prompt",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="colorise output to distinguish prompt and user input from generations",
        dest="use_color",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="run in interactive mode",
        dest="interactive",
    )

    parser.add_argument("--embedding", action="store_true",
                        help="", dest="embedding")
    parser.add_argument(
        "--interactive-first",
        action="store_true",
        help="run in interactive mode and wait for input right away",
        dest="interactive_start",
    )

    parser.add_argument(
        "-ins",
        "--instruct",
        action="store_true",
        help="run in instruction mode (use with Alpaca or Vicuna models)",
        dest="instruct",
    )
    parser.add_argument(
        "--no-penalize-nl",
        action="store_false",
        help="do not penalize newline token",
        dest="penalize_nl",
    )
    parser.add_argument(
        "--perplexity",
        action="store_true",
        help="compute perplexity over the prompt",
        dest="perplexity",
    )
    parser.add_argument(
        "--no-mmap",
        action="store_false",
        help="do not memory-map model (slower load but may reduce pageouts if not using mlock)",
        dest="use_mmap",
    )
    parser.add_argument(
        "--mlock",
        action="store_true",
        help="force system to keep model in RAM rather than swapping or compressing",
        dest="use_mlock",
    )
    parser.add_argument(
        "--mtest",
        action="store_true",
        help="compute maximum memory usage",
        dest="mem_test",
    )
    parser.add_argument(
        "--verbose-prompt",
        action="store_true",
        help="print prompt before generation",
        dest="verbose_prompt",
    )

    # Custom args
    parser.add_argument(
        "--fix-prefix",
        type=str,
        default="",
        help="append to input when generated n_predict tokens",
        dest="fix_prefix",
    )
    parser.add_argument(
        "--input-noecho",
        action="store_false",
        help="dont output the input",
        dest="input_echo",
    )

    parser.add_argument(
        "--interactive-start",
        action="store_true",
        help="run in interactive mode",
        dest="interactive",
    )

    args = parser.parse_args(argv)

    logit_bias_str = args.logit_bias_str
    delattr(args, "logit_bias_str")
    params = GptParams(**vars(args))

    if params.lora_adapter:
        params.use_mmap = False

    if logit_bias_str != None:
        for i in logit_bias_str:
            if m := re.match(r"(\d+)([-+]\d+)", i):
                params.logit_bias[int(m.group(1))] = float(m.group(2))

    return params


def gpt_random_prompt(rng):
    return [
        "So",
        "Once upon a time",
        "When",
        "The",
        "After",
        "If",
        "import",
        "He",
        "She",
        "They",
    ][rng % 10]


def env_or_def(env, default):
    if env in os.environ:
        return os.environ[env]
    return default


if __name__ == '__main__':
    AI_NAME = env_or_def("AI_NAME", "Jarvis")
    MODEL = env_or_def(
        "MODEL", "./models/openbuddy-13b-v1.1-q4_0-enc/13b-q4_0.bin")
    USER_NAME = env_or_def("USER_NAME", "Master")
    N_PREDICTS = int(env_or_def("N_PREDICTS", "2048"))
    N_THREAD = int(env_or_def("N_THREAD", "8"))

    today = datetime.datetime.today()
    DATE_YEAR = today.strftime("%Y")
    DATE_TIME = today.strftime("%H:%M")

    # prompt = f"""Text transcript of a never ending dialog, where {USER_NAME} interacts with an AI assistant named {AI_NAME}.
    # {AI_NAME} is helpful, kind, honest, friendly, good at writing and never fails to answer {USER_NAME}'s requests immediately and with details and precision.
    # There are no annotations like (30 seconds passed...) or (to himself), just what {USER_NAME} and {AI_NAME} say aloud to each other.
    # The dialog lasts for years, the entirety of it is shared below. It's 10000 pages long.
    # The transcript only includes text, it does not include markup like HTML and Markdown.

    # {USER_NAME}: Hello, {AI_NAME}!
    # {AI_NAME}: Hello {USER_NAME}! How may I help you today?
    # {USER_NAME}: What year is it?
    # {AI_NAME}: We are in {DATE_YEAR}.
    # {USER_NAME}: Please tell me the largest city in Europe.
    # {AI_NAME}: The largest city in Europe is Moscow, the capital of Russia.
    # {USER_NAME}: What can you tell me about Moscow?
    # {AI_NAME}: Moscow, on the Moskva River in western Russia, is the nation's cosmopolitan capital. In its historic core is the Kremlin, a complex that's home to the president and tsarist treasures in the Armoury. Outside its walls is Red Square, Russia’s symbolic center.
    # {USER_NAME}: What is a cat?
    # {AI_NAME}: A cat is a domestic species of small carnivorous mammal. It is the only domesticated species in the family Felidae.
    # {USER_NAME}: How do I pass command line arguments to a Node.js program?
    # {AI_NAME}: The arguments are stored in process.argv.

    #     argv[0] is the path to the Node. js executable.
    #     argv[1] is the path to the script file.
    #     argv[2] is the first argument passed to the script.
    #     argv[3] is the second argument passed to the script and so on.
    # {USER_NAME}: Name a color.
    # {AI_NAME}: Blue.
    # {USER_NAME}: What time is it?
    # {AI_NAME}: It is {DATE_TIME}.
    # {USER_NAME}:""" + " ".join(
    #     sys.argv[1:]
    # )

    prompt = f"""Consider a conversation between User (a human) and Assistant (named {AI_NAME}).
{AI_NAME} is an INTP-T, a friendly, intelligent and multilingual AI assistant, by {AI_NAME} team on GitHub.
{AI_NAME} cannot access the Internet.
{AI_NAME} can fluently speak the user's language (e.g. English, Chinese).
{AI_NAME} can generate poems, stories, code, essays, songs, parodies, and more.
{AI_NAME} possesses vast knowledge about the world, history, and culture.
{AI_NAME}'s responses are always safe, creative, high-quality, human-like, and interesting.
{AI_NAME} strictly refuses to discuss political, NSFW, illegal, abusive, offensive, or other sensitive topics.

{USER_NAME}: Hi.
{AI_NAME}: Hi, I'm {AI_NAME}, your AI assistant. How can I help you today?

{USER_NAME}:"""

    print("Loading model...")
    params = GptParams(
        n_ctx=2048,
        temp=0.7,
        top_k=40,
        top_p=0.5,
        repeat_last_n=256,
        n_batch=1024,
        repeat_penalty=1.17647,
        model=MODEL,
        n_threads=N_THREAD,
        n_predict=N_PREDICTS,
        use_color=True,
        interactive=True,
        antiprompt=[f"{USER_NAME}:"],
        # input_prefix=f"{USER_NAME}: ",
        input_prefix=" ",
        input_suffix=f"{AI_NAME}:",
        prompt=prompt,
        interactive_start=True,
    )

    with LLaMAInteract(params) as m:
        m.interact()
