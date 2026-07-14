import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

prompt = "A futuristic city at sunset with flying cars, ultra realistic"

image = pipe(prompt).images[0]

image.save("generated_image.png")

print("Image generated successfully!")