"""
해당 스크립트는 Tokenizer 모델
"""
from transformers import CLIPTokenizer
import triton_python_backend_utils as pb_utils
import numpy as np


class TritonPythonModel:
    def initialize(self, args):
        ...

    def execute(self, requests):
        ...

# https://blog.ml6.eu/triton-ensemble-model-for-deploying-transformers-into-production-c0f727c012e3

