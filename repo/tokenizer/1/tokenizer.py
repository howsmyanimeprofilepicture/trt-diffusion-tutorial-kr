"""
해당 스크립트는 Tokenizer 모델
"""
from transformers import CLIPTokenizer
import triton_python_backend_utils as pb_utils
import numpy as np


class TritonPythonModel:
    def initialize(self, args):
        # 다음과 같이 토크나이저를 로드해줍니다.
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "segmind/tiny-sd",
            subfolder="tokenizer",
            use_fast=True,
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            # request로부터 "prompts"라는 이름의 인풋을 불러옵니다.
            prompts: list[list[bytes]] = (   
                pb_utils.get_input_tensor_by_name(request, "prompts").as_numpy().tolist()
            )   
            prompts: list[str] = [pr[0].decode() for pr in prompts]
            batch_size = len(prompts)
            # 이렇게 불러온 프롬프트를 토크나이징하고,
            prompt_tokens = self.tokenizer(
                prompts,
                return_tensors="np",
                padding="max_length",
                max_length=77,
            )  # (batch_size, 77)
            # Unconditional한 denoising을 위해 빈 텍스트를 토크나이징 합니다.
            uncond_tokens = self.tokenizer(
                batch_size * [""],
                return_tensors="np",
                padding="max_length",
                max_length=77,
            )  # (batch_size,77)

            input_ids = np.stack([prompt_tokens.input_ids, uncond_tokens.input_ids],
                                axis=1).astype(np.int32)
            attention_mask = np.stack([prompt_tokens.attention_mask,
                                       uncond_tokens.attention_mask], 
                                       axis=1).astype(np.int32)  # (batch_size, 2, 77) int64
            input_ids = pb_utils.Tensor("input_ids", input_ids)
            attention_mask = pb_utils.Tensor("attention_mask",
                                             attention_mask)
            #  sample, timestep, i
            sample = pb_utils.Tensor("sample",
                                     np.random.randn(batch_size, 4, 64, 64).astype(np.float32))
            timestep = pb_utils.Tensor("timestep",
                                       np.array([1000], dtype=np.int32))
            i = pb_utils.Tensor("i", np.array([0], dtype=np.uint8))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[input_ids, attention_mask, sample, timestep, i]
            )
            responses.append(inference_response)

        return responses


# https://blog.ml6.eu/triton-ensemble-model-for-deploying-transformers-into-production-c0f727c012e3

