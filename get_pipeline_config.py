

def get_unet_scheduler(i: int):
    return f'''
        {{
            model_name: "unet",
            model_version: -1,
            input_map: [
                {{ key: "sample", value: "sample{i}" }},
                {{ key: "timestep", value: "timestep{i}" }},
                {{ key: "encoder_hidden_states", value: "encoder_hidden_states"}}
            ],
            output_map: [{{ key: "noise", value: "noise{i}" }}]
        }},
        {{
            model_name: "scheduler",
            model_version: -1,
            input_map: [
                {{ key: "sample", value: "sample{i}" }},
                {{ key: "noise", value: "noise{i}" }},
                {{ key: "i", value: "i{i}" }}
            ],
            output_map: [
                {{ key: "sample", value: "sample{i+1}" }},
                {{ key: "timestep", value: "timestep{i+1}" }},
                {{ key: "i", value: "i{i+1}" }}
            ]
        }},
'''


def pbtxt(num_infer: int = 25, img_size: int = 512):

    _unet_schedulers = "\n".join([get_unet_scheduler(i) 
                                  for i in range(0, num_infer-2)])

    return f'''name: "pipeline"
max_batch_size: 0
platform: "ensemble"
input: [
    {{ name: "prompts", data_type: TYPE_STRING, dims: [-1, 1] }}
]
output: [
    {{
        name: "output",
        data_type: TYPE_FP32,
        dims: [-1, 3, {img_size}, {img_size}]    
    }}
]
ensemble_scheduling: {{
    step: [
        {{
            model_name: "tokenizer",
            model_version: -1,
            input_map: [{{ key: "prompts", value: "prompts" }}],
            output_map: [
                {{ key: "input_ids", value: "input_ids" }},
                {{ key: "attention_mask", value: "attention_mask" }},
                {{ key: "sample", value: "sample" }},
                {{ key: "timestep", value: "timestep"  }},
                {{ key: "i",  value: "i"}}
            ]
        }},
        {{
            model_name: "text_encoder",
            model_version: -1,
            input_map: [
                {{ key: "input_ids", value: "input_ids" }},
                {{ key: "attention_mask", value: "attention_mask" }}
            ],
            output_map: [{{ key: "encoder_hidden_states", value: "encoder_hidden_states"}}]
        }},
        {{
            model_name: "unet",
            model_version: -1,
            input_map: [
                {{ key: "sample", value: "sample" }},
                {{ key: "timestep", value: "timestep" }},
                {{ key: "encoder_hidden_states", value: "encoder_hidden_states"}}    
            ],
            output_map: [{{ key: "noise", value: "noise" }}]
        }},
        {{
            model_name: "scheduler",
            model_version: -1,
            input_map: [
                {{ key: "sample", value: "sample" }},
                {{ key: "noise", value: "noise" }},
                {{ key: "i", value: "i" }}
            ],
            output_map: [
                {{ key: "sample", value: "sample0" }},
                {{ key: "timestep", value: "timestep0" }},
                {{ key: "i", value: "i0" }}
            ]
        }},
        {_unet_schedulers}
        {{
            model_name: "unet",
            model_version: -1,
            input_map: [
                {{ key: "sample", value: "sample{num_infer-2}" }},
                {{ key: "timestep", value: "timestep{num_infer-2}" }},
                {{ key: "encoder_hidden_states", value: "encoder_hidden_states"}}
            ],
            output_map: [{{ key: "noise", value: "noise{num_infer-2}" }}]
        }},
        {{
            model_name: "scheduler",
            model_version: -1,
            input_map: [
                {{ key: "sample", value: "sample{num_infer-2}" }},
                {{ key: "noise", value: "noise{num_infer-2}" }},
                {{ key: "i", value: "i{num_infer-2}" }}
            ],
            output_map: [
                {{ key: "sample", value: "sample{num_infer-1}" }}
            ]
        }},
        {{
            model_name: "vae",
            model_version: -1,
            input_map: [{{ key: "z", value: "sample{num_infer-1}" }}],
            output_map: [{{ key: "image", value: "output" }}]
        }}
    ]
}}
'''


if __name__ == "__main__":
    with open("./repo/pipeline/config.pbtxt", "w", encoding="utf-8") as f:
        f.write(pbtxt(25, 512))
    