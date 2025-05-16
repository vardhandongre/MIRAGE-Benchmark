# src/llm_agents.py

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration class for LLM clients"""
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 60
    retries: int = 3

class BaseLLMClient(ABC):
    """Abstract base class for all LLM clients"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    def generate(self, messages: List[Dict], **kwargs) -> str:
        """Generate a response from the LLM"""
        pass
    
    def _merge_config(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge default config with overrides"""
        params = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
        }
        params.update(kwargs)
        return params
    
    def _has_image_content(self, messages: List[Dict]) -> bool:
        """Check if any message contains image content"""
        for msg in messages:
            if isinstance(msg['content'], list):
                for item in msg['content']:
                    if item.get('type') == 'image_url':
                        return True
        return False
    
    def _extract_text_from_messages(self, messages: List[Dict]) -> List[Dict]:
        """Extract text-only messages from potentially multimodal messages"""
        text_messages = []
        for msg in messages:
            if isinstance(msg['content'], list):
                text_parts = []
                for item in msg['content']:
                    if item['type'] == 'text':
                        text_parts.append(item['text'])
                if text_parts:
                    msg_copy = msg.copy()
                    msg_copy['content'] = ' '.join(text_parts)
                    text_messages.append(msg_copy)
            else:
                text_messages.append(msg)
        return text_messages
    
    def _process_image_url(self, image_url: str) -> str:
        """Process image URL and return base64 encoded image data"""
        import base64
        import requests
        from io import BytesIO
        
        if image_url.startswith('data:image'):
            # Extract base64 data from data URI
            return image_url.split(',')[1]
        else:
            # Download image and convert to base64
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                return base64.b64encode(response.content).decode('utf-8')
            except Exception as e:
                self.logger.error(f"Failed to download image: {str(e)}")
                raise ValueError(f"Failed to process image URL: {image_url}")

    def _validate_messages(self, messages: List[Dict]) -> List[Dict]:
        """Validate and format messages for the LLM"""
        valid_roles = {'system', 'user', 'assistant'}
        
        for msg in messages:
            if 'role' not in msg:
                raise ValueError(f"Message missing 'role' field: {msg}")
            if 'content' not in msg:
                raise ValueError(f"Message missing 'content' field: {msg}")
            if msg['role'] not in valid_roles:
                raise ValueError(f"Invalid role '{msg['role']}'. Must be one of: {valid_roles}")
            
            # Validate content can be string or list (for image support)
            if not isinstance(msg['content'], (str, list)):
                raise ValueError(f"Message content must be string or list for multimodal content")
            
            # If content is a list, validate each item
            if isinstance(msg['content'], list):
                for item in msg['content']:
                    if not isinstance(item, dict):
                        raise ValueError(f"Content list items must be dictionaries")
                    if 'type' not in item:
                        raise ValueError(f"Content item missing 'type' field")
                    if item['type'] not in ['text', 'image_url']:
                        raise ValueError(f"Invalid content type '{item['type']}'. Must be 'text' or 'image_url'")
        
        return messages

class OpenAIClient(BaseLLMClient):
    """OpenAI client for GPT models with image support"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import openai
            self.client = openai
            self.client.api_key = os.getenv("OPENAI_API_KEY")
            if not self.client.api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
    
    def generate(self, messages: List[Dict], **kwargs) -> str:
        validated_messages = self._validate_messages(messages)
        params = self._merge_config(kwargs)
        
        # Check if model supports vision (e.g., gpt-4-vision-preview)
        has_images = self._has_image_content(validated_messages)
        if has_images and "vision" not in self.config.model.lower():
            self.logger.warning(f"Model '{self.config.model}' may not support images. Consider using gpt-4-vision-preview or gpt-4o")
        
        try:
            response = self.client.ChatCompletion.create(
                model=self.config.model,
                messages=validated_messages,
                **params
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise

class AnthropicClient(BaseLLMClient):
    """Anthropic client for Claude models with image support"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import anthropic
            self.client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        except ImportError:
            raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
    
    def generate(self, messages: List[Dict], **kwargs) -> str:
        validated_messages = self._validate_messages(messages)
        
        # Convert OpenAI message format to Anthropic format
        system_messages = [msg['content'] for msg in validated_messages if msg['role'] == 'system' and isinstance(msg['content'], str)]
        non_system_messages = []
        
        for msg in validated_messages:
            if msg['role'] != 'system':
                # Anthropic uses a different format for multimodal content
                if isinstance(msg['content'], list):
                    content_list = []
                    for item in msg['content']:
                        if item['type'] == 'text':
                            content_list.append({"type": "text", "text": item['text']})
                        elif item['type'] == 'image_url':
                            # Anthropic expects base64 images
                            image_data = self._process_image_url(item['image_url']['url'])
                            content_list.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",  # Or detect from image
                                    "data": image_data
                                }
                            })
                    msg_copy = msg.copy()
                    msg_copy['content'] = content_list
                    non_system_messages.append(msg_copy)
                else:
                    non_system_messages.append(msg)
        
        try:
            if system_messages:
                response = self.client.messages.create(
                    model=self.config.model,
                    system="\n".join(system_messages),
                    messages=non_system_messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
            else:
                response = self.client.messages.create(
                    model=self.config.model,
                    messages=non_system_messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
            return response.content[0].text.strip()
        except Exception as e:
            self.logger.error(f"Anthropic API error: {str(e)}")
            raise

class TogetherAIClient(BaseLLMClient):
    """TogetherAI client for various models with image support"""

    MODEL_ID_MAP = {
         "meta-llama-3-2-90b-instruct": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
         "meta-llama-3-3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
         "meta-llama-3-2-11b": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
         "meta-llama-4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
         "meta-llama-4-scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
         "qwen72b": "Qwen/Qwen2-VL-72B-Instruct",
         "deepseek-r1": "deepseek-ai/DeepSeek-R1",
         "deepseek-v3": "deepseek-ai/DeepSeek-V3",
         "gemma-3": "google/gemma-3-27b-it",
         "deepseek-judge": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
    }
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import together
            # Set the API key directly on the together module
            together.api_key = os.getenv("TOGETHER_API_KEY")
            if not together.api_key:
                raise ValueError("TOGETHER_API_KEY environment variable not set")
        except ImportError:
            raise ImportError("Together package not installed. Install with: pip install together")
    
    def generate(self, messages: List[Dict], **kwargs) -> str:
        validated_messages = self._validate_messages(messages)
        params = self._merge_config(kwargs)
        
        # Get proper model name from the map
        model_name = self.MODEL_ID_MAP.get(self.config.model, self.config.model)
        
        # Format messages for Together's API
        formatted_messages = self._format_for_together(validated_messages)
        
        try:
            # Import the Together module directly in this method
            import together
            
            # Use the together module's direct API functions
            response = together.Complete.create(
                model=model_name,
                prompt=self._format_messages_as_prompt(formatted_messages),
                temperature=params.get("temperature", 0.3),
                max_tokens=params.get("max_tokens", 1024),
                top_p=params.get("top_p", 1.0)
            )
            
            # Extract response text
            return response['choices'][0]['text'].strip()
        except Exception as e:
            self.logger.error(f"TogetherAI API error: {str(e)}")
            
            # Try alternative API format for vision models based on your example
            try:
                import together
                
                # Use the format shown in your example
                response = together.chat.completions.create(
                    model=model_name,
                    messages=formatted_messages,
                    temperature=params.get("temperature", 0.3),
                    max_tokens=params.get("max_tokens", 1024),
                    top_p=params.get("top_p", 1.0),
                    stream=False
                )
                
                # Extract response text based on the example response format
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    return response.choices[0].message.content.strip()
                else:
                    # Fallback to dict access if attribute access fails
                    return response['choices'][0]['message']['content'].strip()
            except Exception as nested_e:
                self.logger.error(f"Alternative TogetherAI API error: {str(nested_e)}")
                raise
    
    def _format_for_together(self, messages: List[Dict]) -> List[Dict]:
        """Format messages for Together's API, handling images correctly"""
        formatted_messages = []
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            # For multimodal messages, format properly
            if isinstance(content, list):
                formatted_content = []
                
                for item in content:
                    if item['type'] == 'text':
                        formatted_content.append({"type": "text", "text": item['text']})
                    elif item['type'] == 'image_url':
                        # Format image URL according to Together's expected format
                        formatted_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": item['image_url']['url']
                            }
                        })
                
                formatted_messages.append({"role": role, "content": formatted_content})
            else:
                # For text-only messages
                formatted_messages.append({"role": role, "content": content})
        
        return formatted_messages
        
    def _format_messages_as_prompt(self, messages: List[Dict]) -> str:
        """Convert messages to a text prompt format for older API versions"""
        prompt = ""
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            # Handle different message formats
            if isinstance(content, list):
                # For multimodal messages, extract text and add image placeholders
                text_parts = []
                for item in content:
                    if item['type'] == 'text':
                        text_parts.append(item['text'])
                    elif item['type'] == 'image_url':
                        text_parts.append("[IMAGE]")  # Placeholder for image
                content_text = " ".join(text_parts)
            else:
                content_text = content
            
            # Format based on role
            if role == "system":
                prompt += f"System: {content_text}\n\n"
            elif role == "user":
                prompt += f"User: {content_text}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content_text}\n\n"
        
        # Add final prompt for the assistant to continue
        prompt += "Assistant:"
        
        return prompt

class BedrockClient(BaseLLMClient):
    """AWS Bedrock client for various models with image support"""
    
    # Mapping of friendly names to Bedrock model IDs
    MODEL_ID_MAP = {
        # Claude models
        "claude-3-7-sonnet": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude-3-5-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
        "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
        "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
        "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        
        # Nova models
        "nova-pro": "us.amazon.nova-pro-v1:0",
        "nova-premier": "amazon.nova-premier-v1:0",
        "nova-lite": "amazon.nova-lite-v1:0",
        "nova-micro": "amazon.nova-micro-v1:0",
        
        # Llama models
        "llama3-1-405b-instruct": "us.meta.llama3-1-405b-instruct-v1:0",
        "llama3-1-70b-instruct": "us.meta.llama3-1-70b-instruct-v1:0",
        "llama3-1-8b-instruct": "us.meta.llama3-1-8b-instruct-v1:0",
        "llama3-3-70b-instruct": "us.meta.llama3-3-70b-instruct-v1:0",
        "llama3-2-11b-instruct": "meta.llama3-2-11b-instruct-v1:0",
        "llama3-2-3b-instruct": "meta.llama3-2-3b-instruct-v1:0",
        "llama3-2-1b-instruct": "meta.llama3-2-1b-instruct-v1:0",
        
        # Titan models
        "titan-text-g1-premier": "amazon.titan-text-premier-v1:0",
        "titan-text-g1-express": "amazon.titan-text-express-v1",
        "titan-image-generator-g1": "amazon.titan-image-generator-v1",
        
        # Mistral models
        "mistral-large": "mistral.mistral-large-2407-v1:0",
        "mistral-7b-instruct": "mistral.mistral-7b-instruct-v0:2",
        "mixtral-8x7b-instruct": "mistral.mixtral-8x7b-instruct-v0:1",
        
        # Cohere models
        "command-r-plus": "cohere.command-r-plus-v1:0",
        "command-r": "cohere.command-r-v1:0",
        "command": "cohere.command-text-v14",
        "command-light": "cohere.command-light-text-v14",
        
        # Stability AI models
        "stable-diffusion-xl": "stability.stable-diffusion-xl-v1",
        "stable-diffusion-xl-light": "stability.stable-diffusion-xl-v0",
        "stable-image-core": "stability.stable-image-core-v1:0",
        "stable-image-ultra": "stability.stable-image-ultra-v1:0",
    }
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import boto3
            self.bedrock = boto3.client('bedrock-runtime', 
                region_name=os.getenv("AWS_REGION", "us-east-1"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )
        except ImportError:
            raise ImportError("boto3 package not installed. Install with: pip install boto3")
    
    def _get_model_id(self, model_name: str) -> str:
        """Get the actual Bedrock model ID from a friendly name"""
        # If the model_name looks like a full model ID (contains dots), return as-is
        if '.' in model_name:
            return model_name
        
        # Look up the model ID in our mapping
        model_id = self.MODEL_ID_MAP.get(model_name)
        if model_id is None:
            # Try case-insensitive lookup
            for key, value in self.MODEL_ID_MAP.items():
                if key.lower() == model_name.lower():
                    return value
            
            # If not found, raise an error with available models
            available_models = ', '.join(sorted(self.MODEL_ID_MAP.keys()))
            raise ValueError(
                f"Unknown model name: '{model_name}'. "
                f"Available models: {available_models}"
            )
        
        return model_id
    
    def generate(self, messages: List[Dict], **kwargs) -> str:
        validated_messages = self._validate_messages(messages)
        params = self._merge_config(kwargs)
        
        # Resolve model name to actual model ID
        model_id = self._get_model_id(self.config.model)
        
        # Check if model supports vision
        has_images = self._has_image_content(validated_messages)
        if has_images and not self._model_supports_vision(model_id):
            raise ValueError(f"Model '{model_id}' does not support vision. Use Claude 3 or newer.")
        
        # Format depends on the model being used
        body = self._format_for_bedrock(validated_messages, params)
        
        try:
            response = self.bedrock.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            return self._extract_response(response_body)
        except Exception as e:
            self.logger.error(f"Bedrock API error: {str(e)}")
            raise
    
    def _format_for_bedrock(self, messages: List[Dict], params: Dict) -> Dict:
        """Format messages for specific Bedrock models with image support"""
        model_id = self._get_model_id(self.config.model)
        
        if 'anthropic.claude' in model_id:
            # Format for Claude models (supports multimodal)
            anthropic_messages = []
            for msg in messages:
                if msg['role'] != 'system':
                    if isinstance(msg['content'], list):
                        content_list = []
                        for item in msg['content']:
                            if item['type'] == 'text':
                                content_list.append({"type": "text", "text": item['text']})
                            elif item['type'] == 'image_url':
                                # Bedrock expects base64 images for Claude
                                image_data = self._process_image_url(item['image_url']['url'])
                                content_list.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": image_data
                                    }
                                })
                        anthropic_messages.append({
                            "role": msg['role'],
                            "content": content_list
                        })
                    else:
                        anthropic_messages.append({
                            "role": msg['role'],
                            "content": msg['content']
                        })
            
            body = {
                "messages": anthropic_messages,
                "max_tokens": params.get("max_tokens", 1024),
                "temperature": params.get("temperature", 0.3),
                "top_p": params.get("top_p", 1.0),
                "anthropic_version": "bedrock-2023-05-31"
            }
            
            # Add system message if present
            system_msgs = [msg['content'] for msg in messages if msg['role'] == 'system' and isinstance(msg['content'], str)]
            if system_msgs:
                body["system"] = "\n".join(system_msgs)
            
            return body
        elif 'amazon.titan-text' in model_id:
            # Titan models don't support multimodal - extract text only
            text_parts = []
            for msg in messages:
                if isinstance(msg['content'], list):
                    for item in msg['content']:
                        if item['type'] == 'text':
                            text_parts.append(item['text'])
                        elif item['type'] == 'image_url':
                            text_parts.append(f"[Image: {item['image_url']['url'][:50]}...]")
                else:
                    text_parts.append(msg['content'])
            
            return {
                "inputText": "\n".join(text_parts),
                "textGenerationConfig": {
                    "maxTokenCount": params.get("max_tokens", 1024),
                    "temperature": params.get("temperature", 0.3),
                    "topP": params.get("top_p", 1.0),
                }
            }
        elif 'amazon.nova' in model_id:
            # Nova models use a different format
            return {
                "messages": messages,
                "maxTokens": params.get("max_tokens", 1024),
                "temperature": params.get("temperature", 0.3),
                "topP": params.get("top_p", 1.0)
            }
        elif 'meta.llama' in model_id:
            # Llama models on Bedrock
            prompt = self._format_messages_for_llama(messages)
            return {
                "prompt": prompt,
                "max_gen_len": params.get("max_tokens", 1024),
                "temperature": params.get("temperature", 0.3),
                "top_p": params.get("top_p", 1.0),
            }
        else:
            # Default format for other models
            return {
                "messages": messages,
                "maxTokens": params.get("max_tokens", 1024),
                "temperature": params.get("temperature", 0.3)
            }
    
    def _extract_response(self, response_body: Dict) -> str:
        """Extract text from Bedrock response based on model type"""
        # Handle Claude 3 response format (Messages API)
        if 'content' in response_body and isinstance(response_body['content'], list):
            for item in response_body['content']:
                if item.get('type') == 'text':
                    return item['text'].strip()
        
        # Handle older Claude format
        if 'completion' in response_body:
            return response_body['completion'].strip()
        
        # Handle newer Claude format with content array
        if 'message' in response_body and 'content' in response_body['message']:
            for item in response_body['message']['content']:
                if item.get('type') == 'text':
                    return item['text'].strip()
        
        # Handle Titan format
        if 'results' in response_body:
            return response_body['results'][0]['outputText'].strip()
        
        # Handle Llama format
        if 'generation' in response_body:
            return response_body['generation'].strip()
        
        # Handle Nova format
        if 'output' in response_body and 'choices' in response_body['output']:
            choices = response_body['output']['choices']
            if choices and 'message' in choices[0]:
                return choices[0]['message']['content'].strip()
        
        # Handle Meta Llama formats
        if 'outputs' in response_body and response_body['outputs']:
            output = response_body['outputs'][0]
            if 'text' in output:
                return output['text'].strip()
            elif 'content' in output:
                return output['content'].strip()
        
        # If none of the expected formats match, log the structure
        self.logger.error(f"Unknown response format: {response_body}")
        raise ValueError(f"Unexpected Bedrock response format. Content: {response_body}")
    
    def _model_supports_vision(self, model_id: str) -> bool:
        """Check if the model supports vision capabilities"""
        vision_models = [
            "anthropic.claude-3-haiku",
            "anthropic.claude-3-sonnet",
            "anthropic.claude-3-opus",
            "anthropic.claude-3-5-sonnet",
            "anthropic.claude-3-5-haiku",
            "anthropic.claude-3-7-sonnet",
            "us.anthropic.claude-3-7-sonnet",
            "amazon.nova-pro",
            "amazon.nova-premier",
        ]
        return any(model in model_id.lower() for model in vision_models)
    
    def _format_messages_for_llama(self, messages: List[Dict]) -> str:
        """Format messages for Llama models"""
        formatted = ""
        for msg in messages:
            role = msg['role']
            if isinstance(msg['content'], list):
                content = self._extract_text_from_content(msg['content'])
            else:
                content = msg['content']
            
            if role == "system":
                formatted += f"<|system|>\n{content}</s>\n"
            elif role == "user":
                formatted += f"<|user|>\n{content}</s>\n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content}</s>\n"
        
        # Add assistant prompt for generation
        formatted += "<|assistant|>\n"
        return formatted
    
    def _extract_text_from_content(self, content: List[Dict]) -> str:
        """Extract text from multimodal content list"""
        text_parts = []
        for item in content:
            if item['type'] == 'text':
                text_parts.append(item['text'])
            elif item['type'] == 'image_url':
                text_parts.append(f"[Image: {item['image_url']['url'][:30]}...]")
        return " ".join(text_parts)

class LLMFactory:
    """Factory class for creating LLM clients"""
    
    SUPPORTED_PROVIDERS = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "together": TogetherAIClient,
        "bedrock": BedrockClient,
    }
    
    @classmethod
    def create(cls, provider: str, model: str, **config_kwargs) -> BaseLLMClient:
        """Create an LLM client based on provider and model"""
        if provider not in cls.SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: {list(cls.SUPPORTED_PROVIDERS.keys())}")
        
        config = LLMConfig(model=model, **config_kwargs)
        client_class = cls.SUPPORTED_PROVIDERS[provider]
        
        try:
            return client_class(config)
        except Exception as e:
            logger.error(f"Failed to create {provider} client: {str(e)}")
            raise

# Simple interface function for backward compatibility
def get_llm_client(provider: str, model: str, **kwargs) -> BaseLLMClient:
    """Get an LLM client with optional configuration parameters"""
    return LLMFactory.create(provider, model, **kwargs)

# Utility functions for common operations
def batch_generate(client: BaseLLMClient, message_batches: List[List[Dict]], **kwargs) -> List[str]:
    """Generate responses for multiple message batches"""
    responses = []
    for messages in message_batches:
        try:
            response = client.generate(messages, **kwargs)
            responses.append(response)
        except Exception as e:
            logger.error(f"Batch generation failed: {str(e)}")
            responses.append(f"Error: {str(e)}")
    return responses

def stream_generate(client: BaseLLMClient, messages: List[Dict], **kwargs) -> Optional[str]:
    """Generate with streaming support (if available)"""
    # This would need to be implemented differently for each provider
    # This is a placeholder for future streaming implementation
    return client.generate(messages, **kwargs)

# Example usage
if __name__ == "__main__":
    # Create a client
    llm = get_llm_client("openai", "gpt-4-vision-preview", temperature=0.5)
    
    # Example 1: Text-only conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    try:
        response = llm.generate(messages)
        print(f"Text Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Multimodal conversation with image
    multimodal_messages = [
        {"role": "system", "content": "You are a helpful assistant that can see and interpret images."},
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": "What do you see in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg",
                        "detail": "high"  # Optional: used by some providers
                    }
                }
            ]
        }
    ]
    
    try:
        response = llm.generate(multimodal_messages)
        print(f"Multimodal Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Using different providers with image support
    providers = {
        "openai": "gpt-4-vision-preview",
        "anthropic": "claude-3-opus",
        "bedrock": "anthropic.claude-3-sonnet-20240229-v1:0"
    }
    
    for provider, model in providers.items():
        try:
            print(f"\nTesting {provider} with model {model}")
            client = get_llm_client(provider, model)
            
            # Test with multimodal message
            test_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,/9j/4AAQSkZJRg...base64data...",
                            }
                        }
                    ]
                }
            ]
            
            response = client.generate(test_messages)
            print(f"{provider} Response: {response[:100]}...")
            
        except Exception as e:
            print(f"{provider} Error: {e}")
    
    # Example 4: Using with base64 images from file
    def encode_image_file(image_path: str) -> str:
        """Encode local image file to base64"""
        import base64
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    # Test with local image
    local_image_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What objects are in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image_file('path/to/image.jpg')}"
                    }
                }
            ]
        }
    ]
    
    try:
        response = llm.generate(local_image_messages)
        print(f"Local Image Response: {response}")
    except Exception as e:
        print(f"Local Image Error: {e}")
    
    # Example 5: Batch processing with images
    batch_messages = [
        [
            {"role": "user", "content": "What's 2+2?"}
        ],
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Count the objects in this image"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/objects.jpg"}}
                ]
            }
        ],
        [
            {"role": "user", "content": "What's the capital of Japan?"}
        ]
    ]
    
    try:
        batch_responses = batch_generate(llm, batch_messages)
        for i, resp in enumerate(batch_responses):
            print(f"Batch {i+1}: {resp[:50]}...")
    except Exception as e:
        print(f"Batch Error: {e}")