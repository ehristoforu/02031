import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from transformers import T5EncoderModel, CLIPTextModel, CLIPTokenizer, T5TokenizerFast
import modules.model_management as model_management

from diffusers import FlowMatchEulerDiscreteScheduler, DPMSolverMultistepScheduler

def set_scheduler(pipe, scheduler_name):
    if scheduler_name == 'FlowMatchEulerDiscreteScheduler':
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == 'DPMSolverMultistepScheduler':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe

import os

def load_flux_pipe(model_path, dtype=torch.bfloat16, quant_config=None, gguf_path=None):
    """
    Loads the FLUX pipeline with optional quantization from either a directory or a single file.
    """
    pipe_kwargs = {"torch_dtype": dtype}
    
    # Handle single file loading (.safetensors or .gguf)
    if os.path.isfile(model_path):
        if model_path.endswith(".safetensors"):
            print(f"Loading FLUX pipeline from single file: {model_path}")
            pipe = FluxPipeline.from_single_file(model_path, **pipe_kwargs)
        elif model_path.endswith(".gguf"):
            print(f"Loading GGUF transformer from: {model_path}")
            from diffusers import GGUFQuantizationConfig
            quantization_config = GGUFQuantizationConfig(compute_dtype=dtype)
            transformer = FluxTransformer2DModel.from_single_file(
                model_path,
                quantization_config=quantization_config,
                torch_dtype=dtype,
            )
            pipe_kwargs["transformer"] = transformer
            # Load other components from the default model
            pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", **pipe_kwargs)
        else:
            raise ValueError("Unsupported single file format. Please use .safetensors or .gguf")
    
    # Handle directory loading (diffusers format)
    else:
        transformer = None
        text_encoder_2 = None

        if gguf_path:
            from diffusers import GGUFQuantizationConfig
            print(f"Loading GGUF transformer from: {gguf_path}")
            quantization_config = GGUFQuantizationConfig(compute_dtype=dtype)
            transformer = FluxTransformer2DModel.from_single_file(
                gguf_path,
                quantization_config=quantization_config,
                torch_dtype=dtype,
            )

        if quant_config:
            from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
            from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig

            print(f"Applying quantization: {quant_config}")
            if quant_config['load_in_8bit']:
                transformer_quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
                t5_quant_config = TransformersBitsAndBytesConfig(load_in_8bit=True)
            elif quant_config['load_in_4bit']:
                transformer_quant_config = DiffusersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
                t5_quant_config = TransformersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
            
            if not transformer:
                transformer = FluxTransformer2DModel.from_pretrained(
                    model_path, subfolder="transformer", quantization_config=transformer_quant_config, torch_dtype=dtype
                )
            
            text_encoder_2 = T5EncoderModel.from_pretrained(
                model_path, subfolder="text_encoder_2", quantization_config=t5_quant_config, torch_dtype=dtype
            )

        if transformer:
            pipe_kwargs["transformer"] = transformer
        if text_encoder_2:
            pipe_kwargs["text_encoder_2"] = text_encoder_2
            
        pipe = FluxPipeline.from_pretrained(model_path, **pipe_kwargs)

    pipe.enable_model_cpu_offload()
    return pipe

def generate_flux(pipe, prompt, negative_prompt="", width=1024, height=1024, num_inference_steps=50, guidance_scale=3.5, seed=0):
    """
    Generates an image using the FLUX pipeline.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    # Note: FLUX does not use a negative prompt in the same way as SD.
    # We pass it for consistency, but it might not be used by the standard pipeline.
    image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    
    return image