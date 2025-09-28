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

def load_flux_pipe(model_path, dtype=torch.bfloat16):
    """
    Loads the FLUX pipeline from a directory or a single safetensors file.
    """
    pipe_kwargs = {"torch_dtype": dtype}
    
    if os.path.isfile(model_path) and model_path.endswith(".safetensors"):
        print(f"Loading FLUX pipeline from single file: {model_path}")
        pipe = FluxPipeline.from_single_file(model_path, **pipe_kwargs)
    elif os.path.isdir(model_path):
        print(f"Loading FLUX pipeline from directory: {model_path}")
        pipe = FluxPipeline.from_pretrained(model_path, **pipe_kwargs)
    else:
        raise ValueError(f"Unsupported model path: {model_path}. Please provide a directory or a .safetensors file.")

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