import base64
import io
from PIL import Image
from openai import OpenAI
from langchain_community.llms import Ollama
import time

class ImageProcessor:
    def __init__(self, model_choice="gpt-4o-mini", api_key=None):
        self.model_choice = model_choice
        if api_key and model_choice == "gpt-4o-mini":
            self.openai_client = OpenAI(api_key=api_key)
        # Initialize Ollama with the correct model name
        self.ollama_client = Ollama(
            model="llava",  # Using llava model which is better for vision tasks
            timeout=60,  # Increased timeout for better processing
        )
    
    def encode_image_to_base64(self, image):
        """Optimize and encode image"""
        try:
            # Resize image if too large
            max_size = 1024
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save as PNG for better quality
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            raise Exception(f"Error encoding image: {str(e)}")
    
    def process_image(self, image):
        """Process image using selected model with retry logic"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                if self.model_choice == "gpt-4o-mini":
                    return self.process_image_gpt4(image)
                else:
                    return self.process_image_llama(image)
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Error processing image after {max_retries} attempts: {str(e)}"
                time.sleep(retry_delay * (attempt + 1))
    
    def process_image_gpt4(self, image):
        """Process image using GPT-4 Vision"""
        if not hasattr(self, 'openai_client'):
            return "Error: OpenAI API key not provided for GPT-4 Vision"
        
        try:
            base64_image = self.encode_image_to_base64(image)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this image in detail. Describe any text, diagrams, visual elements, and their relationships. If there are any technical concepts or mathematical formulas, explain them clearly."},
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
        except Exception as e:
            return f"Error with GPT-4 Vision: {str(e)}"
    
    def process_image_llama(self, image):
        """Process image using Llama Vision"""
        try:
            img_str = self.encode_image_to_base64(image)
            
            # Shorter, more focused prompt for faster processing
            prompt = """Analyze this image concisely:
1. Main visual elements and relationships
2. Key text or diagrams and their significance
3. Technical concepts 
Be brief but specific."""
            
            # Optimized Ollama settings
            response = self.ollama_client.invoke(
                prompt,
                images=[img_str],
                stream=False,
                options={
                    "temperature": 0.1,
                    "num_ctx": 2048,     # Reduced context window
                    "num_predict": 256,   # Shorter responses
                    "stop": ["##", "```"] # Stop tokens for shorter responses
                }
            )
            
            if not response or not isinstance(response, str):
                return "Error: Invalid response from Llama model"
            
            return response.strip()
            
        except Exception as e:
            if "timeout" in str(e).lower():
                return "Error: Model timed out. Please try again."
            return f"Error with Llama: {str(e)}"