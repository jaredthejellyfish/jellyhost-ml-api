import cv2
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, UploadFile
from fastapi import FastAPI, File, UploadFile
from diffusers import StableDiffusionPipeline
from fastapi.responses import StreamingResponse


app = FastAPI()

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")


@app.get("/")
async def root():
    return {"response": "Hello, welcome to this stable diffusion API!"}


@app.get("/generate")
async def generate(prompt: str, inference_steps: int = 50, guideance_scale: float = 7.5, negative_prompt: str = None, height: int = 568, width: int = 568):
    image = pipe(prompt, num_inference_steps=inference_steps, guidance_scale=guideance_scale, negative_prompt=negative_prompt,
                 height=height, width=width).images[0]

    hires_img = BytesIO()
    image.save(hires_img, "PNG")
    hires_img.seek(0)

    return StreamingResponse(content=hires_img, media_type="image/png")


@app.post("/upscale")
def image_filter(image: UploadFile = File(...), upscaler: str = "edsr", scale: int = 2):

    upscaler_f_name = f"models/{upscaler}_{scale}.pb"

    original_image = Image.open(image.file)

    lowres_img = np.array(original_image)

    super_res = cv2.dnn_superres.DnnSuperResImpl_create()

    super_res.readModel(upscaler_f_name)
    super_res.setModel(upscaler, scale)
    espcn_image = super_res.upsample(lowres_img)

    lowres_img = Image.fromarray(espcn_image)

    hires_img = BytesIO()
    lowres_img.save(hires_img, "PNG")
    hires_img.seek(0)

    return StreamingResponse(hires_img, media_type="image/png")
