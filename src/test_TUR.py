import torch
from torch import autocast
import diffusers
import datetime

import diffusers.schedulers
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDPMScheduler

YOUR_TOKEN = "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"

def gen_images(pipe,name:str,num_images=10,init_seed=0,num_inference_steps=50):
    for i in range(num_images):
        #seed固定
        SEED=i+init_seed
        generator = torch.Generator(device=DEVICE).manual_seed(SEED) 
        #print(pipe.scheduler)
        with autocast(DEVICE): 
            image = pipe(prompt, guidance_scale=7.5, num_inference_steps=num_inference_steps,generator=generator)["images"][0] 
        # 現在時間 
        dt_now = datetime.datetime.now() 
        now = dt_now.strftime("%Y%m%d%H%M%S") 
        # ファイル名 
        file_path = name+str(SEED) + "_" + str(now)+"_"+str(i)+str(num_inference_steps) + ".png"
        # ファイル保存 
        image.save(file_path)  


MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID,                                         
                                               revision="fp16", 
                                               torch_dtype=torch.float16, 
                                               use_auth_token=YOUR_TOKEN,
                                               debug=True)
pipe.to(DEVICE)

prompt = "a dog painted by Katsuhika Hokusai"

#### schedulerの変更 ###
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
gen_images(pipe,"euler_discrete",10,num_inference_steps=500)



