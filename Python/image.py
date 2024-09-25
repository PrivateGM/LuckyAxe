import torch
from diffusers import StableDiffusionPipeline

print(torch.__version__)

# Загрузка модели
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Убедитесь, что у вас есть доступ к GPU

def generate_image(prompt):
    with torch.autocast("cuda"):
        image = pipe(prompt).images[0]
    image.save(f"{prompt}.png")
    print(f"Изображение '{prompt}.png' создано!")

if __name__ == "__main__":
    user_prompt = input("Введите запрос для генерации изображения: ")
    generate_image(user_prompt)
