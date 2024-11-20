from typing import List, Dict, Optional, Union
import os
from PIL import Image
import anthropic
import time
import copy 
import json
import base64
import io

# Configure the Claude API
client = anthropic.Anthropic(
    api_key=os.getenv("CLAUDE_API_KEY")
)

def resize_image_to_limit(image_path: str, max_size_bytes: int = 4 * 1024 * 1024) -> bytes:  # Reduced to 4MB for safety
    """
    Resize image to be under max_size_bytes while maintaining aspect ratio.
    Returns bytes of the resized JPEG image.
    """
    try:
        # Open image
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # Initial quality and scale
        quality = 90  # Start with slightly lower quality
        scale = 1.0
        
        while True:
            # Create a bytes buffer
            buffer = io.BytesIO()
            
            # Resize image if scale changed
            if scale < 1.0:
                new_size = tuple(int(dim * scale) for dim in img.size)
                resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
            else:
                resized_img = img
            
            # Save to buffer with progressive JPEG
            resized_img.save(buffer, 
                           format='JPEG', 
                           quality=quality, 
                           optimize=True,
                           progressive=True)
            
            # Get base64 size
            base64_size = len(base64.b64encode(buffer.getvalue()))
            
            # Check if base64 size is under limit
            if base64_size <= max_size_bytes:
                buffer.seek(0)
                return buffer.getvalue()
                
            # If still too big, adjust parameters more aggressively
            if quality > 30:
                quality -= 15  # More aggressive quality reduction
            else:
                scale *= 0.7  # More aggressive size reduction

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        raise

def verify_base64_size(base64_str: str, max_size_bytes: int = 5 * 1024 * 1024) -> bool:
    """Verify that base64 string is under size limit"""
    size = len(base64_str.encode('utf-8'))
    return size <= max_size_bytes

def encode_image(image_path: str) -> str:
    """Encode image to base64 string, ensuring it's under 5MB."""
    try:
        # Resize image and get bytes
        image_bytes = resize_image_to_limit(image_path)
        base64_str = base64.b64encode(image_bytes).decode('utf-8')
        
        # Verify final size
        if not verify_base64_size(base64_str):
            raise ValueError(f"Image still too large after resizing: {len(base64_str.encode('utf-8'))} bytes")
            
        return base64_str
    except Exception as e:
        print(f"Warning: Failed to process image {image_path}: {e}")
        raise

class Claude:
    def __init__(self, model_name='claude-3-5-sonnet-latest', messages=[]):
        self.model_name = model_name
        self.messages = []  # Store only user/assistant messages
        self.system = ""    # Store system message separately
        
        # Initialize messages, separating system from other messages
        for msg in messages:
            if msg["role"] == "system":
                self.system = msg["content"]
            else:
                self.messages.append(msg)
                
        self.history = copy.deepcopy(messages)
        self.max_retries = 5

        # Token usage tracking
        self.prompt_tokens = 0
        self.completion_tokens = 0
        
    def chat(self, prompt: str, images: List[str] = [], response_format=None) -> str:
        """
        Send a message to Claude and get a response.
        
        Args:
            prompt (str): The text prompt to send
            images (List[str]): List of image paths to include
            
        Returns:
            str: The response from Claude as a JSON string
        """
        # Prepare message content
        if not images:
            self.messages.append({"role": "user", "content": prompt})
            self.history.append({"role": "user", "content": prompt})
        else:
            content = [{"type": "text", "text": prompt}]
            successful_images = []
            
            for image in images:
                try:
                    base64_image = encode_image(image)
                    # Double-check size before adding
                    if verify_base64_size(base64_image):
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image
                            }
                        })
                        successful_images.append(image)
                    else:
                        print(f"Warning: Skipping image {image} as it's still too large after resizing")
                except Exception as e:
                    print(f"Warning: Skipping image {image} due to error: {e}")
                    continue
            
            if successful_images:
                self.messages.append({"role": "user", "content": content})
                self.history.append({
                    "role": "user", 
                    "content": prompt,
                    "processed_images": successful_images
                })
            else:
                print("Warning: No images were successfully processed")
                self.messages.append({"role": "user", "content": prompt})
                self.history.append({"role": "user", "content": prompt})

        # Configure message parameters
        message_params = {
            "model": self.model_name,
            "messages": self.messages,
            "max_tokens": 8192,
        }
        
        # Add system message if present
        if self.system:
            message_params["system"] = self.system

        # Attempt to get response with retries
        for attempt in range(self.max_retries):
            try:
                response = client.messages.create(**message_params)
                
                # Update token counts
                self.prompt_tokens += response.usage.input_tokens
                self.completion_tokens += response.usage.output_tokens
                
                # Extract response content
                response_content = response.content[0].text
                
                # Add response to message history
                self.messages.append({
                    "role": "assistant",
                    "content": response_content
                })
                self.history.append({
                    "role": "assistant",
                    "content": response_content
                })
                
                return response_content
                
            except anthropic.APIError as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"Failed after {self.max_retries} attempts. Error: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
                
    def get_history(self) -> List[Dict]:
        """Return the conversation history."""
        return self.history
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.messages = []
        self.system = ""
        self.history = []
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def info(self) -> Dict:
        """Return token usage information."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens
        }