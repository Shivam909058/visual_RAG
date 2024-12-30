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
                model="gpt-4-vision-preview",
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
            # Convert image to base64
            img_str = self.encode_image_to_base64(image)
            
            # Create a detailed prompt for better image analysis
            prompt = """Please analyze this image in detail and provide:
1. A description of all visual elements present
2. Any text content and its meaning
3. Explanation of diagrams or technical illustrations if present
4. Relationships between different elements in the image
5. Any mathematical formulas or technical concepts shown
6. The main message or purpose of this image

Provide your analysis in a clear, structured format."""
            
            # Call Ollama with the enhanced prompt
            response = self.ollama_client.invoke(
                prompt,
                images=[img_str],
                stream=False,
                options={
                    "temperature": 0.1,  # Lower temperature for more focused responses
                    "num_ctx": 4096,     # Larger context window
                    "num_predict": 512   # More detailed responses
                }
            )
            
            # Check and format response
            if not response or not isinstance(response, str):
                return "Error: Invalid response from Llama model"
            
            # Clean up and structure the response
            cleaned_response = response.strip()
            if not cleaned_response:
                return "Error: Empty response from Llama model"
                
            return cleaned_response
            
        except Exception as e:
            if "timeout" in str(e).lower():
                return "Error: Llama model timed out. Please try again."
            return f"Error processing with Llama: {str(e)}"