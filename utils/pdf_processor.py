import fitz
from PIL import Image
import io

class PDFProcessor:
    @staticmethod
    def extract_text_and_images(pdf_file):
        """Extract both text and images from PDF"""
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        images = []
        image_locations = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            text += page_text
            
            # Extract images
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(image_bytes))
                images.append(image)
                
                # Store image location within the text
                image_locations.append({
                    'page': page_num + 1,
                    'text_before': page_text[:1000]  # Store some surrounding text
                })
        
        return text, images, image_locations 