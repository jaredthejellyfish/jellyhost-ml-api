FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /bot
COPY . /bot/

RUN pip install -r requirements.txt

RUN apt update && apt install -y git curl libgl1-mesa-glx libgl1-mesa-dri && pip install --upgrade git+https://github.com/huggingface/diffusers.git

EXPOSE 9568

ENTRYPOINT ["uvicorn" ,"api:app", "--port", "9568", "--host", "0.0.0.0"]
