import argparse
import gc
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

from diffusers import OnnxStableDiffusionPipeline, OnnxStableDiffusionImg2ImgPipeline
from diffusers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, EulerDiscreteScheduler
from diffusers import __version__ as _df_version
import gradio as gr
import numpy as np
from packaging import version
import PIL

is_v_0_4 = version.parse(_df_version) >= version.parse("0.4.0") # Negative prompt
is_v_0_6 = version.parse(_df_version) >= version.parse("0.6.0") # Img2Img
is_v_0_7 = version.parse(_df_version) >= version.parse("0.7.0") # Inpainting
is_v_0_8 = version.parse(_df_version) >= version.parse("0.8.0") # Inpainting without finetuned model

if is_v_0_7:
    from diffusers import OnnxStableDiffusionInpaintPipeline

if is_v_0_8:
    from diffusers import OnnxStableDiffusionInpaintPipelineLegacy


def get_latents_from_seed(seed: int, batch_size: int, height: int, width: int) -> np.ndarray:
    latents_shape = (batch_size, 4, height // 8, width // 8)
    rng = np.random.default_rng(seed)
    image_latents = rng.standard_normal(latents_shape).astype(np.float32)
    return image_latents


def run_diffusers(
    prompt: str,
    neg_prompt: str,
    init_image: PIL.Image.Image,
    iteration_count: int,
    batch_size: int,
    steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    eta: float,
    denoise_strength: float,
    seed: str,
    mask_image: PIL.Image.Image
) -> Tuple[list, str]:
    global model_name
    global current_pipe
    global pipe

    prompt.strip("\n")
    neg_prompt.strip("\n")

    if seed == "":
        rng = np.random.default_rng()
        seed = rng.integers(np.iinfo(np.uint32).max)
    else:
        try:
            seed = int(seed) & np.iinfo(np.uint32).max
        except ValueError:
            seed = hash(seed) & np.iinfo(np.uint32).max
    seeds = np.array([seed], dtype=np.uint32)
    if iteration_count > 1:
        seed_seq = np.random.SeedSequence(seed)
        seeds = np.concatenate((seeds, seed_seq.generate_state(iteration_count - 1)))

    output_path = "output"
    os.makedirs(output_path, exist_ok=True)

    sched_name = str(pipe.scheduler._class_name)
    prompts = [prompt]*batch_size
    neg_prompts = [neg_prompt]*batch_size if neg_prompt != "" else None
    images = []
    time_taken = 0
    output_base_path = Path("./output")
    for i in range(iteration_count):
        dt_obj = datetime.now()
        dt_cust = dt_obj.strftime("%Y-%m-%d_%H-%M-%S")
        image_name = dt_cust + "_" + str(seed) + ".png"
        text_name = dt_cust + "_" + str(seed) + "_info.txt"
        image_path = output_base_path / image_name
        text_path = output_base_path / text_name
        info = f"Prompt: {prompt}\nNegative prompt: {neg_prompt}\nSeed: {seeds[i]}\n" + \
            f"Scheduler: {sched_name}\nScale: {guidance_scale}\nHeight: {height}\nWidth: {width}\nETA: {eta}\n" + \
            f"Model: {model_name}\nIteration size: {iteration_count}\nBatch size: {batch_size}\nSteps: {steps}"
        if (current_pipe == "img2img" or current_pipe == "inpaint" ):
            info = info + f" denoise: {denoise_strength}"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(info)

        if current_pipe == "txt2img":
            # Generate our own latents so that we can provide a seed.
            latents = get_latents_from_seed(seeds[i], batch_size, height, width)

            start = time.time()
            batch_images = pipe(
                prompts, negative_prompt=neg_prompts, height=height, width=width, num_inference_steps=steps,
                guidance_scale=guidance_scale, eta=eta, latents=latents).images
            finish = time.time()
        elif current_pipe == "img2img":
            # NOTE: at this time there's no good way of setting the seed for the random noise added by the scheduler
            # np.random.seed(seeds[i])
            start = time.time()
            batch_images = pipe(
                prompts, negative_prompt=neg_prompts, init_image=init_image, height=height, width=width,
                num_inference_steps=steps, guidance_scale=guidance_scale, eta=eta, strength=denoise_strength,
                num_images_per_prompt=batch_size).images
            finish = time.time()
        elif current_pipe == "inpaint":
            start = time.time()
            # NOTE: legacy require init_image but inpainting use image
            batch_images = pipe(
                prompts, negative_prompt=neg_prompts, image=init_image, mask_image=mask_image, height=height, width=width,
                num_inference_steps=steps, guidance_scale=guidance_scale, eta=eta, strength=denoise_strength,
                num_images_per_prompt=batch_size, init_image=init_image).images
            finish = time.time()

        short_prompt = prompt.strip("<>:\"/\\|?*\n\t")
        short_prompt = short_prompt[:99] if len(short_prompt) > 100 else short_prompt
        for j in range(batch_size):
            batch_images[j].save(image_path)

        images.extend(batch_images)
        time_taken = time_taken + (finish - start)

    time_taken = time_taken / 60.0
    if iteration_count > 1:
        status = f"Run took {time_taken:.1f} minutes " + \
            f"to generate {iteration_count} iterations with batch size of {batch_size}. seeds: " + \
            np.array2string(seeds, separator=",")
    else:
        status = f"Run took {time_taken:.1f} minutes to generate a batch size of " + \
            f"{batch_size}. seed: {seeds[0]}"

    return images, status


def clear_click():
    return {
        prompt: "", neg_prompt: "", image: None, mask: None, mask_mode: "Draw mask", sch: "Euler", iter: 1, batch: 1,
        drawn_mask: None, steps: 25, guid: 11, height: 512, width: 512, eta: 0.0, denoise: 0.8, seed: ""}


def generate_click(
    model_drop, prompt, neg_prompt, sch, iter, batch, steps, guid, height, width, eta,
    seed, image, denoise, mask, pipeline, mask_mode, drawn_mask
):
    global model_name
    global provider
    global current_pipe
    global scheduler
    global pipe
    
    # reset scheduler and pipeline if model is different
    if model_name != model_drop:
        model_name = model_drop
        scheduler = None
        pipe = None
    model_path = os.path.join("model", model_name)

    if sch == "PNDM" and type(scheduler) is not PNDMScheduler:
        scheduler = PNDMScheduler.from_config(model_path, subfolder="scheduler")
    elif sch == "LMS" and type(scheduler) is not LMSDiscreteScheduler:
        scheduler = LMSDiscreteScheduler.from_config(model_path, subfolder="scheduler")
    elif sch == "DDIM" and type(scheduler) is not DDIMScheduler:
        scheduler = DDIMScheduler.from_config(model_path, subfolder="scheduler")
    elif sch == "Euler" and type(scheduler) is not EulerDiscreteScheduler:
        scheduler = EulerDiscreteScheduler.from_config(model_path, subfolder="scheduler")

    # select which pipeline depending on current tab
    if pipeline == "TEXT2IMG":
        if current_pipe != "txt2img" or pipe is None:
            pipe = OnnxStableDiffusionPipeline.from_pretrained(
                model_path, provider=provider, scheduler=scheduler)
            pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
            gc.collect()
        current_pipe = "txt2img"

        if type(pipe.scheduler) is not type(scheduler):
            pipe.scheduler = scheduler

        return run_diffusers(
            prompt, neg_prompt, None, iter, batch, steps, guid, height, width, eta, 0,
            seed, None)
    elif pipeline == "IMG2IMG":
        if current_pipe != "img2img" or pipe is None:
            pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(
                model_path, provider=provider, scheduler=scheduler)
            pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
            gc.collect()
        current_pipe = "img2img"

        if type(pipe.scheduler) is not type(scheduler):
            pipe.scheduler = scheduler

        # input image resizing
        input_image = image.convert("RGB")
        input_width, input_height = input_image.size
        if height / width > input_height / input_width:
            adjust_width = int(input_width * height / input_height)
            input_image = input_image.resize((adjust_width, height))
            left = (adjust_width - width) // 2
            right = left + width
            input_image = input_image.crop((left, 0, right, height))
        else:
            adjust_height = int(input_height * width / input_width)
            input_image = input_image.resize((width, adjust_height))
            top = (adjust_height - height) // 2
            bottom = top + height
            input_image = input_image.crop((0, top, width, bottom))

        return run_diffusers(
            prompt, neg_prompt, input_image, iter, batch, steps, guid, height, width, eta,
            denoise, seed, None)
    elif pipeline == "Inpainting" and is_v_0_7:
        if current_pipe != "inpaint" or pipe is None:
            # >=0.8.0: Model name must ends with "inpainting" to use the proper pipeline
            # This allows usage of Legacy pipeline for models not finetuned for inpainting
            # see huggingface/diffusers!51
            if is_v_0_8 and not model_name.endswith("inpainting"):
                pipe = OnnxStableDiffusionInpaintPipelineLegacy.from_pretrained(
                    model_path, provider=provider, scheduler=scheduler)
            else:
                # on >=0.7.0 & <0.8.0 or model finetuned for inpainting
                pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained(
                    model_path, provider=provider, scheduler=scheduler)
            pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
            gc.collect()
        current_pipe = "inpaint"

        if type(pipe.scheduler) is not type(scheduler):
            pipe.scheduler = scheduler

        if mask_mode == "Upload mask":
            input_image = image.convert("RGB")

            # input mask resizing
            input_mask = mask.convert("RGB")
            input_mask_width, input_mask_height = input_mask.size
            if height / width > input_mask_height / input_mask_width:
                adjust_mask_width = int(input_mask_width * height / input_mask_height)
                input_mask = input_mask.resize((adjust_mask_width, height))
                mask_left = (adjust_mask_width - width) // 2
                mask_right = mask_left + width
                input_mask = input_mask.crop((mask_left, 0, mask_right, height))
            else:
                adjust_height = int(input_mask_height * width / input_mask_width)
                input_mask = input_mask.resize((width, adjust_height))
                top = (adjust_height - height) // 2
                bottom = top + height
                input_mask = input_mask.crop((0, top, width, bottom))
        else:
            input_image = drawn_mask['image'].convert('RGB')
            input_mask = drawn_mask['mask']

        # input image resizing
        input_width, input_height = input_image.size
        if height / width > input_height / input_width:
            adjust_width = int(input_width * height / input_height)
            input_image = input_image.resize((adjust_width, height))
            left = (adjust_width - width) // 2
            right = left + width
            input_image = input_image.crop((left, 0, right, height))
        else:
            adjust_height = int(input_height * width / input_width)
            input_image = input_image.resize((width, adjust_height))
            top = (adjust_height - height) // 2
            bottom = top + height
            input_image = input_image.crop((0, top, width, bottom))


        return run_diffusers(
            prompt, neg_prompt, input_image, iter, batch, steps, guid, height, width, eta,
            denoise, seed, input_mask)


def choose_sch(sched_name: str):
    if sched_name == "DDIM":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def choose_pipeline(pipeline: str, mask_mode: str):
    if(pipeline == "TEXT2IMG"):
        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
    if(pipeline == "IMG2IMG"):
        return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False))
    if(pipeline == "Inpainting"):
        if mask_mode == "Draw mask":
            return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True))
        else:
            return (gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False))

def choose_mask_mode(mask_mode):
    if mask_mode == "Draw mask":
        return [gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)]
    else:
        return [gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)]

def size_512_lock(size):
    if size != 512:
        return gr.update(interactive=False)
    return gr.update(interactive=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradio interface for ONNX based Stable Diffusion")
    parser.add_argument("--cpu-only", action="store_true", default=False, help="Run ONNX with CPU")
    parser.add_argument("--local", action="store_true", default=False, help="Open to local network")
    parser.add_argument("--public", action="store_true", default=False, help="Create a publicly shareable link for the interface")
    args = parser.parse_args()

    # variables for ONNX pipelines
    model_name = None
    provider = "CPUExecutionProvider" if args.cpu_only else "DmlExecutionProvider"
    current_pipe = "txt2img"

    # diffusers objects
    scheduler = None
    pipe = None

    # search the model folder
    model_dir = "model"
    model_list = []
    with os.scandir(model_dir) as scan_it:
        for entry in scan_it:
            if entry.is_dir():
                model_list.append(entry.name)
    default_model = model_list[0] if len(model_list) > 0 else None

    # create gradio block
    title = "Stable Diffusion " + str(version.parse(_df_version))
    possibilities = ['TEXT2IMG']
    if is_v_0_6:
        possibilities.append('IMG2IMG')
    if is_v_0_7:
        possibilities.append('Inpainting')
    with gr.Blocks(title=title) as app:
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                with gr.Column(variant='panel'):
                    with gr.Row():
                        model_drop = gr.Dropdown(model_list, value=default_model, label="Model", interactive=True)
                        pipeline = gr.Radio(possibilities, value="TEXT2IMG", label="Pipeline")
                    sch = gr.Radio(["DDIM", "LMS", "PNDM", "Euler"], value="Euler", label="Scheduler")
                    eta = gr.Slider(0, 1, value=0.0, step=0.01, label="DDIM eta", visible=False)
                    seed = gr.Textbox(value="", max_lines=1, label="Seed")
                with gr.Column():
                    mask_mode = gr.Radio(label="Mask mode", show_label=False, choices=["Draw mask", "Upload mask"], value="Draw mask", visible=False)
                    image = gr.Image(label="Input image", type="pil", visible=False)
                    mask = gr.Image(label="Input mask", type="pil", visible=False)
                    drawn_mask = gr.Image(label="Input image and mask", source="upload", tool="sketch", type="pil", visible=False)
                    prompt = gr.Textbox(value="", lines=2, label="Prompt")
                    neg_prompt = gr.Textbox(value="", lines=2, label="Negative prompt", visible=is_v_0_4)
                    steps = gr.Slider(1, 150, value=25, step=1, label="Steps")
                    guid = gr.Slider(0, 20, value=11, step=0.5, label="Guidance")
                with gr.Column():
                    height = gr.Slider(16, 1920, value=512, step=8, label="Height")
                    width = gr.Slider(16, 1920, value=512, step=8, label="Width")
                    denoise = gr.Slider(0, 1, value=0.8, step=0.01, label="Denoise strength", visible=False)
                with gr.Column():
                    iter = gr.Slider(1, 24, value=1, step=1, label="Iteration count")
                    batch = gr.Slider(1, 4, value=1, step=1, label="Batch size")
            with gr.Column(scale=1, min_width=600):
                with gr.Row():
                    gen_btn = gr.Button("Generate", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")
                with gr.Row(variant='panel'):
                    image_out = gr.Gallery(value=None, label="Images")
                status_out = gr.Textbox(value="", label="Status")

        # config components
        all_inputs = [
            model_drop, prompt, neg_prompt, sch, iter, batch, steps, guid, height, width,
            eta, seed, image, denoise, mask, pipeline, mask_mode, drawn_mask]
        clear_btn.click(fn=clear_click, inputs=None, outputs=all_inputs, queue=False)
        gen_btn.click(fn=generate_click, inputs=all_inputs, outputs=[image_out, status_out])

        height.change(fn=size_512_lock, inputs=height, outputs=width)
        width.change(fn=size_512_lock, inputs=width, outputs=height)
        mask_mode.change(fn=choose_mask_mode, inputs=mask_mode, outputs=[image, mask, drawn_mask])
        pipeline.change(fn=choose_pipeline, inputs=[pipeline, mask_mode], outputs=[image, mask, denoise, mask_mode, drawn_mask])
        sch.change(fn=choose_sch, inputs=sch, outputs=eta)

        image_out.style(grid=2)

    app.queue(concurrency_count=1, api_open=False)
    app.launch(inbrowser=True, server_name="0.0.0.0" if args.local else "127.0.0.1", show_api=False, quiet=True, share=args.public) # open to local network: server_name="0.0.0.0"
