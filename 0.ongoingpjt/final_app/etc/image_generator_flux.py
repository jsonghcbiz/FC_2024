import torch
from diffusers import FluxPipeline
import os
from PIL import Image
import time

class ImageGenerator:
    def __init__(self):
        # Initialize the FLUX pipeline
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", 
            torch_dtype=torch.bfloat16
        )
        # Enable CPU offload to save VRAM
        self.pipe.enable_model_cpu_offload()

    def generate_image(self, prompt, num_images=1):
        """
        Generate images based on the given prompt using FLUX.1-dev
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs('generated_images', exist_ok=True)
            
            # Generate images
            images = self.pipe(
                prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
                num_images_per_prompt=num_images,
                generator=torch.Generator("cpu").manual_seed(0)
            ).images
            
            # Save images
            saved_paths = []
            for i, image in enumerate(images):
                timestamp = int(time.time())
                filename = f"generated_images/flux_image_{timestamp}_{i}.png"
                image.save(filename)
                saved_paths.append(filename)
                print(f"Image saved as {filename}")
            
            return saved_paths
            
        except Exception as e:
            print(f"Error generating image: {e}")
            return None

# Example usage
if __name__ == "__main__":
    generator = ImageGenerator()
    
    # Example prompt for movie poster
    test_prompt = """Create a cinematic movie poster with dramatic lighting, 
    professional composition, high contrast, movie title text, 
    theatrical release style"""
    
    generator.generate_image(test_prompt, num_images=1) 