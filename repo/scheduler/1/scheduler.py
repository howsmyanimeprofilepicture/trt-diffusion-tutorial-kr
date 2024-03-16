import triton_python_backend_utils as pb_utils
import numpy as np

class TritonPythonModel:
    num_train_timesteps = 1000
    num_inference_steps: int = 25
    beta_start: float = 0.00085
    beta_end: float = 0.012
    guidance_scale: float = 7.5
    

    def initialize(self, args):
        self.betas = (
            np.linspace(self.beta_start ** (0.5),
                        self.beta_end ** (0.5),
                        self.num_train_timesteps + 1) ** 2
        )
        self.alphas = 1. - self.betas
        self.alphas_cumprod = self.alphas.cumprod(axis=0)
        self.timesteps = np.linspace(
            self.num_train_timesteps - self.num_train_timesteps % self.num_inference_steps,
            0,
            self.num_inference_steps + 1,
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            (sample, noise, i) = self.get_inputs(request)
            i = i[0]
            sample = sample.astype(np.float64)
            noise = noise.astype(np.float64)

           
            (t, prev_t) = self.timesteps[i: i+2].astype(np.int32)
            alpha_prod_t = self.alphas_cumprod[t]  # <- float64
            alpha_prod_t_prev = self.alphas_cumprod[prev_t]
            beta_prod_t = 1 - alpha_prod_t
            current_alpha_t = alpha_prod_t / alpha_prod_t_prev
            current_beta_t = 1 - current_alpha_t

            updated_sample = (
                sample - noise * current_beta_t / (beta_prod_t**0.5)
            )/(current_alpha_t**0.5)

            new_sampled_noise = np.random.randn(1, 4, 64, 64)
            updated_sample = updated_sample + (
                self._get_sigma2(t, prev_t)**.5
            ) * new_sampled_noise


            updated_sample = updated_sample.astype(np.float32)

            updated_sample = pb_utils.Tensor("sample",
                                             updated_sample)

            timestep = pb_utils.Tensor("timestep",
                                       prev_t[None, ...])

            next_i = pb_utils.Tensor("i",
                                     np.array([i+1], dtype=np.uint8))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    updated_sample, timestep, next_i
                ]
            )
            responses.append(inference_response)

        return responses

    def _get_sigma2(self, t: int, prev_t: int):
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t]
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t
        sigma2 = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        return sigma2

    def get_inputs(self, request) -> tuple[np.ndarray,
                                           np.ndarray,
                                           int]:
        return [
            pb_utils
            .get_input_tensor_by_name(request, name)
            .as_numpy()

            for name in ["sample", "noise", "i"]
        ]



