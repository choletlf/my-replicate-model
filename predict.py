from cog import BasePredictor, Input, Path
from diffusers import DiffusionPipeline
import torch

class Predictor(BasePredictor):
    def setup(self):
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.float16,
            safety_checker=None,
            custom_pipeline="lpw_stable_diffusion"
        )
        self.pipe.to("cuda")
        self.pipe.load_lora_weights("XLabs-AI/flux-furry-lora", weight_name="furry_lora.safetensors")

    def predict(self, prompt: str = Input(description="Prompt")) -> Path:
        image = self.pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
        output_path = "/tmp/out.png"
        image.save(output_path)
        return Path(output_path)
