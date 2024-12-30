import base64
import io
from PIL import Image
from openai import OpenAI
from langchain_community.llms import Ollama

class ImageProcessor:
    def __init__(self, model_choice="gpt-4o-mini", api_key=None):
        self.model_choice = model_choice
        if api_key and model_choice == "gpt-4o-mini":
            self.openai_client = OpenAI(api_key=api_key)
        self.ollama_client = Ollama(model="llama3.2-vision")
    
    def encode_image_to_base64(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def process_image(self, image):
        """Process image using selected model"""
        if self.model_choice == "gpt4":
            return self.process_image_gpt4(image)
        else:
            return self.process_image_llama(image)
    
    def process_image_gpt4(self, image):
        """Process image using GPT-4 Vision"""
        if not hasattr(self, 'openai_client'):
            return "Error: OpenAI API key not provided for GPT-4 Vision"
            
        base64_image = self.encode_image_to_base64(image)
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail, including any text, diagrams, or visual elements you can see."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1024
        )
        
        return response.choices[0].message.content
    
    def process_image_llama(self, image):
        """Process image using Llama 2 Vision"""
        try:
            response = self.ollama_client.invoke(
                "Describe this image in detail, including any text, diagrams, or visual elements you can see.",
                images=[image]
            )
            return response
        except Exception as e:
            return f"Error processing with Llama: {str(e)}"