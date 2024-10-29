import argparse, os
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import gradio as gr
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

config = OmegaConf.load(r"configs\latent-diffusion\mydata_ldm-vq-f8_test.yaml")

def parse_args():
    parser=argparse.ArgumentParser(description="Gradio Interface for Image to Image")
    parser.add_argument("--dataset", type=str, help="Dataset path", default=r"my_latent_diffusion_big_dataset\test\outwear")
    return parser.parse_args()

args=parse_args()

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

# Placeholder for your neural network (replace with your actual model)
def load_neural_network():
    model = load_model_from_config(config, "last.ckpt")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)
    return sampler

        
def sample(sketch ,texture, scale_1, scale_2,ddim_steps, ddim_eta=0.0):
    f = 8 # first stage downsampling factor
    model=sampler.model
    c = {
        "sketch":sketch,
        "texture":texture
    }
    cond = model.get_learned_conditioning(c)
    cond = {"cond":cond}
    unconditional_guidance_scale = {
        "scale_1":scale_1,
        "scale_2":scale_2,
    }
    unconditional_conditioning = {
        "two_zero":model.get_learned_conditioning(
        {
            "sketch":torch.zeros_like(c["sketch"],device=c["sketch"].device),
            "texture":torch.zeros_like(c["texture"],device=c["texture"].device)
        }),
        "one_zero":model.get_learned_conditioning(
        {
            "sketch":torch.zeros_like(c["sketch"],device=c["sketch"].device),
            "texture":c["texture"]
        }
    )}

    shape = [4, 256//f, 256//f]

    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                     conditioning=cond,
                                     batch_size=1,
                                     shape=shape,
                                     verbose=False,
                                     unconditional_guidance_scale=unconditional_guidance_scale,
                                     unconditional_conditioning=unconditional_conditioning,
                                     eta=ddim_eta
                                     )

    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
    
    return x_samples_ddim

def transform_img(img):
    img = img.resize((256,256))
    img = np.array(img).astype(np.uint8)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    img = np.transpose(img, (2, 0, 1))
    img = (img/127.5 - 1.0).astype(np.float32)

    return torch.from_numpy(img[np.newaxis, ...]).cuda()

# Your image processing and neural network inference logic
def generate_images(sketch_img_name, texture_img_name, scale_1, scale_2, ddim_steps):
    global args
    
    sketch_img_path = os.path.join(args.dataset, "sketch/", sketch_img_name)
    texture_img_path = os.path.join(args.dataset, "texture/", texture_img_name)
    gt_img_path = os.path.join(args.dataset, "gt/", sketch_img_name.split(".")[0] + ".jpg")


    sketch_img = transform_img(Image.open(sketch_img_path)).cuda()
    texture_img = transform_img(Image.open(texture_img_path)).cuda()
    gt_img = transform_img(Image.open(gt_img_path)).cuda()


    predictions = sample(sketch_img, texture_img,scale_1,scale_2, ddim_steps)
        

    sketch_img = torch.clamp((sketch_img+1.0)/2.0, min=0.0, max=1.0)
    texture_img = torch.clamp((texture_img+1.0)/2.0, min=0.0, max=1.0)
    gt_img = torch.clamp((gt_img+1.0)/2.0, min=0.0, max=1.0)

    return (
            sketch_img.detach().cpu().squeeze().permute(0,1).numpy(),
            texture_img.detach().cpu().squeeze(0).permute(1,2,0).numpy(),
            gt_img.detach().cpu().squeeze(0).permute(1,2,0).numpy(),
            predictions.detach().cpu().squeeze(0).permute(1,2,0).numpy()
        )

# Define the Gradio interface
iface = gr.Interface(
    fn=generate_images,
    inputs=[
        gr.Dropdown(sorted([x for x in os.listdir(args.dataset+"/sketch/")])),
        gr.Dropdown(sorted([x for x in os.listdir(args.dataset+"/texture/")])),
        gr.Slider(minimum=1, maximum=7.5, label="scale_1",value=config["model"]["params"]["scale_1"]),
        gr.Slider(minimum=1, maximum=7.5, label="scale_2",value=config["model"]["params"]["scale_2"]),
        gr.Slider(minimum=50, maximum=200, label="DDIM Step",value=200),
    ],
    outputs=[
        gr.Image(type="numpy", label="sketch",width=256),
        gr.Image(type="numpy", label="texture",width=256),
        gr.Image(type="numpy", label="gt",width=256),
        gr.Image(type="numpy", label="Processed Images",width=256),
    ],
)
if __name__ == "__main__":
    sampler = load_neural_network()

    # Launch the Gradio interface
    iface.launch(server_name="127.0.0.1")
