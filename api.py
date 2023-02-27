import cv2
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, UploadFile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from urllib.parse import unquote
from fastapi import Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

app = FastAPI()


@app.get("/")
async def root():
    return {"response": "Hello, welcome to this machine learning API!"}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )


@app.get("/generate")
async def generate(prompt: str, inference_steps: int = 50, guideance_scale: float = 7.5, negative_prompt: str = None, height: int = 568, width: int = 568, seed: int = None):
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
    pipe.requires_safety_checker = False

    prompt = unquote(prompt)

    if negative_prompt:
        negative_prompt = unquote(negative_prompt)

    if seed:
        seed = int(seed)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        image = pipe(prompt, num_inference_steps=inference_steps, guidance_scale=guideance_scale, negative_prompt=negative_prompt,
                     height=height, width=width, generator=generator).images[0]
    else:
        image = pipe(prompt, num_inference_steps=inference_steps, guidance_scale=guideance_scale, negative_prompt=negative_prompt,
                     height=height, width=width).images[0]

    del pipe

    torch.cuda.empty_cache()

    hires_img = BytesIO()
    image.save(hires_img, "PNG")
    hires_img.seek(0)

    return StreamingResponse(content=hires_img, media_type="image/png")


@app.post("/upscale")
async def upscale(image: UploadFile = File(...), upscaler: str = "edsr", scale: int = 2):
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


@app.post("/img2img")
async def img2img(image: UploadFile = File(...), prompt: str = "", strength: float = 0.8, num_inference_steps: int = 50, guidance_scale: float = 7.5, negative_prompt: str = None):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")

    prompt = unquote(prompt)

    if negative_prompt:
        negative_prompt = unquote(negative_prompt)

    original_image = Image.open(image.file).convert("RGB")
    original_image.thumbnail((720, 720), Image.ANTIALIAS)

    image = pipe(prompt=prompt, image=original_image, strength=strength, guidance_scale=guidance_scale,
                 num_inference_steps=num_inference_steps, negative_prompt=negative_prompt).images[0]

    final_img = BytesIO()
    image.save(final_img, "PNG")
    final_img.seek(0)

    del pipe

    torch.cuda.empty_cache()

    return StreamingResponse(content=final_img, media_type="image/png")
