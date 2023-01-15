# jellyhost-ml-api

This is an API for generating images using Stable Diffusion and upscaling them using a variety of upscalers.

## Usage

To generate an image, send a GET request to the /generate endpoint. The request body should include the prompt you wish to use, as well as the desired inference steps, guidance scale, negative prompt, height, and width for the image.

Example request body:
{
    "prompt": "This is a test.",
    "inference_steps": 50,
    "guidance_scale": 7.5,
    "negative_prompt": None,
    "height": 568,
    "width": 568
}


The response body will contain the generated image, e.g.:
{
    "response": "<image data>"
}


To upscale an existing image, send a POST request to the /upscale endpoint. The request body should include the uploaded image, as well as the desired upscaler and scale.

Example request body:
{
    "image": "<image data>",
    "upscaler": "edsr",
    "scale": 2
}


The response body will contain the upscaled image, e.g.:
{
    "response": "<image data>"
