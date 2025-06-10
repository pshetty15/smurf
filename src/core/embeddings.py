"""
Embedding utilities - extracted from utils.py
Handles Amazon Bedrock embeddings and contextual enrichment
"""
import os
import boto3
import json
import time
from typing import List, Tuple
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Initialize Bedrock client
def get_bedrock_client():
    """Initialize and return Bedrock client with proper configuration."""
    region = os.getenv("AWS_REGION")
    profile_name = os.getenv("AWS_PROFILE")
    endpoint_url = os.getenv("BEDROCK_ENDPOINT_URL")
    
    # Validate required environment variables
    if not region:
        raise ValueError("AWS_REGION environment variable is required")
    if not profile_name:
        raise ValueError("AWS_PROFILE environment variable is required")
    if not endpoint_url:
        raise ValueError("BEDROCK_ENDPOINT_URL environment variable is required")
    
    # Create session with profile from ~/.aws/credentials
    session = boto3.Session(
        profile_name=profile_name,
        region_name=region
    )
    
    return session.client('bedrock-runtime', 
                         region_name=region,
                         endpoint_url=endpoint_url)


def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts using Amazon Bedrock.
    
    Args:
        texts: List of texts to create embeddings for
        
    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        return []
    
    # Get embedding model from environment or use default
    embedding_model = os.getenv("BEDROCK_EMBEDDING_MODEL")
    if not embedding_model:
        raise ValueError("BEDROCK_EMBEDDING_MODEL environment variable is required")
    
    bedrock_client = get_bedrock_client()
    
    max_retries = 3
    retry_delay = 1.0  # Start with 1 second delay
    
    embeddings = []
    
    for retry in range(max_retries):
        try:
            # Bedrock typically processes embeddings one at a time
            # For batch processing, we'll iterate through texts
            batch_embeddings = []
            successful_count = 0
            
            for i, text in enumerate(texts):
                try:
                    # Prepare the request body based on the model
                    if "titan" in embedding_model.lower():
                        body = json.dumps({
                            "inputText": text
                        })
                    elif "cohere" in embedding_model.lower():
                        body = json.dumps({
                            "texts": [text],
                            "input_type": "search_document"
                        })
                    else:
                        # Default to Titan format
                        body = json.dumps({
                            "inputText": text
                        })
                    
                    # Make the API call
                    response = bedrock_client.invoke_model(
                        modelId=embedding_model,
                        body=body,
                        contentType='application/json',
                        accept='application/json'
                    )
                    
                    # Parse response based on model
                    response_body = json.loads(response['body'].read())
                    
                    if "titan" in embedding_model.lower():
                        embedding = response_body['embedding']
                    elif "cohere" in embedding_model.lower():
                        embedding = response_body['embeddings'][0]
                    else:
                        # Try to extract embedding from common response formats
                        if 'embedding' in response_body:
                            embedding = response_body['embedding']
                        elif 'embeddings' in response_body:
                            embedding = response_body['embeddings'][0]
                        else:
                            raise ValueError(f"Unknown response format for model {embedding_model}")
                    
                    batch_embeddings.append(embedding)
                    successful_count += 1
                    
                    # Add small delay between requests to avoid rate limiting
                    if i < len(texts) - 1:
                        time.sleep(0.1)
                        
                except ClientError as e:
                    error_code = e.response['Error']['Code']
                    if error_code == 'ThrottlingException':
                        # If throttled, wait longer and retry
                        if retry < max_retries - 1:
                            print(f"Throttling detected for text {i}, retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2
                            continue
                    print(f"AWS Bedrock error for text {i}: {e}")
                    # Add zero embedding as fallback (adjust size based on your model)
                    embedding_size = 1536 if "titan" in embedding_model.lower() else 1024
                    batch_embeddings.append([0.0] * embedding_size)
                except Exception as e:
                    print(f"Error creating embedding for text {i}: {e}")
                    # Add zero embedding as fallback
                    embedding_size = 1536 if "titan" in embedding_model.lower() else 1024
                    batch_embeddings.append([0.0] * embedding_size)
            
            print(f"Successfully created {successful_count}/{len(texts)} embeddings via Bedrock")
            return batch_embeddings
            
        except Exception as e:
            if retry < max_retries - 1:
                print(f"Error creating batch embeddings (attempt {retry + 1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"Failed to create batch embeddings after {max_retries} attempts: {e}")
                # Return zero embeddings as final fallback
                embedding_size = 1536 if "titan" in embedding_model.lower() else 1024
                return [[0.0] * embedding_size for _ in texts]


def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using Amazon Bedrock.
    
    Args:
        text: Text to create an embedding for
        
    Returns:
        List of floats representing the embedding
    """
    try:
        embeddings = create_embeddings_batch([text])
        return embeddings[0] if embeddings else [0.0] * 1536
    except Exception as e:
        print(f"Error creating embedding: {e}")
        # Return empty embedding if there's an error
        embedding_model = os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v1")
        embedding_size = 1536 if "titan" in embedding_model.lower() else 1024
        return [0.0] * embedding_size


def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.
    Uses Amazon Bedrock for text generation.
    
    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for
        
    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    # Get the text generation model from environment
    text_model = os.getenv("BEDROCK_TEXT_MODEL")
    if not text_model:
        raise ValueError("BEDROCK_TEXT_MODEL environment variable is required")
    
    bedrock_client = get_bedrock_client()
    
    try:
        # Create the prompt for generating contextual information
        prompt = f"""<document> 
{full_document[:25000]} 
</document>
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk}
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        # Prepare request body based on model family
        if "claude" in text_model.lower():
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 200,
                "temperature": 0.3,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
        elif "titan" in text_model.lower():
            body = json.dumps({
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 200,
                    "temperature": 0.3
                }
            })
        else:
            # Default to Claude format
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 200,
                "temperature": 0.3,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })

        # Call Bedrock for text generation
        response = bedrock_client.invoke_model(
            modelId=text_model,
            body=body,
            contentType='application/json',
            accept='application/json'
        )
        
        # Parse response based on model
        response_body = json.loads(response['body'].read())
        
        if "claude" in text_model.lower():
            context = response_body['content'][0]['text'].strip()
        elif "titan" in text_model.lower():
            context = response_body['results'][0]['outputText'].strip()
        else:
            # Try to extract text from common response formats
            if 'content' in response_body and len(response_body['content']) > 0:
                context = response_body['content'][0]['text'].strip()
            elif 'results' in response_body and len(response_body['results']) > 0:
                context = response_body['results'][0]['outputText'].strip()
            else:
                raise ValueError(f"Unknown response format for model {text_model}")
        
        # Combine the context with the original chunk
        contextual_text = f"{context}\n---\n{chunk}"
        
        return contextual_text, True
    
    except Exception as e:
        print(f"Error generating contextual embedding with Bedrock: {e}. Using original chunk instead.")
        return chunk, False


def extract_code_blocks(content: str, min_length: int = 300) -> list:
    """Extract code blocks from markdown content."""
    import re
    
    # Pattern to match code blocks with optional language
    pattern = r'```(\w+)?\n(.*?)\n```'
    matches = re.findall(pattern, content, re.DOTALL)
    
    blocks = []
    for lang, code in matches:
        if len(code.strip()) >= min_length:
            # Find context around the code block
            code_start = content.find(f'```{lang or ""}\n{code}\n```')
            context_before = content[max(0, code_start-500):code_start]
            context_after = content[code_start+len(code)+20:code_start+len(code)+520]
            
            blocks.append({
                'language': lang or 'text',
                'code': code.strip(),
                'context_before': context_before,
                'context_after': context_after
            })
    
    return blocks


def generate_code_example_summary(code: str, context_before: str, context_after: str) -> str:
    """
    Generate a summary for a code example using its surrounding context.
    
    Args:
        code: The code example
        context_before: Context before the code
        context_after: Context after the code
        
    Returns:
        A summary of what the code example demonstrates
    """
    text_model = os.getenv("BEDROCK_TEXT_MODEL")
    if not text_model:
        raise ValueError("BEDROCK_TEXT_MODEL environment variable is required")
    
    bedrock_client = get_bedrock_client()
    
    # Create the prompt
    prompt = f"""<context_before>
{context_before[-500:] if len(context_before) > 500 else context_before}
</context_before>

<code_example>
{code[:1500] if len(code) > 1500 else code}
</code_example>

<context_after>
{context_after[:500] if len(context_after) > 500 else context_after}
</context_after>

Based on the code example and its surrounding context, provide a concise summary (2-3 sentences) that describes what this code example demonstrates and its purpose. Focus on the practical application and key concepts illustrated.
"""
    
    try:
        # Prepare request body based on model family
        if "claude" in text_model.lower():
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "temperature": 0.3,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
        elif "titan" in text_model.lower():
            body = json.dumps({
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 100,
                    "temperature": 0.3
                }
            })
        else:
            # Default to Claude format
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "temperature": 0.3,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })

        # Call Bedrock for text generation
        response = bedrock_client.invoke_model(
            modelId=text_model,
            body=body,
            contentType='application/json',
            accept='application/json'
        )
        
        # Parse response based on model
        response_body = json.loads(response['body'].read())
        
        if "claude" in text_model.lower():
            return response_body['content'][0]['text'].strip()
        elif "titan" in text_model.lower():
            return response_body['results'][0]['outputText'].strip()
        else:
            # Try to extract text from common response formats
            if 'content' in response_body and len(response_body['content']) > 0:
                return response_body['content'][0]['text'].strip()
            elif 'results' in response_body and len(response_body['results']) > 0:
                return response_body['results'][0]['outputText'].strip()
            else:
                return "Code example for demonstration purposes."
        
    except Exception as e:
        print(f"Error generating code example summary with Bedrock: {e}")
        return "Code example for demonstration purposes."


def extract_source_summary(source_id: str, content: str, max_length: int = 500) -> str:
    """
    Extract a summary for a source from its content using Amazon Bedrock.
    
    This function uses the Bedrock API to generate a concise summary of the source content.
    
    Args:
        source_id: The source ID (domain)
        content: The content to extract a summary from
        max_length: Maximum length of the summary
        
    Returns:
        A summary string
    """
    # Default summary if we can't extract anything meaningful
    default_summary = f"Content from {source_id}"
    
    if not content or len(content.strip()) == 0:
        return default_summary
    
    # Get the text generation model from environment
    text_model = os.getenv("BEDROCK_TEXT_MODEL")
    if not text_model:
        raise ValueError("BEDROCK_TEXT_MODEL environment variable is required")
    
    bedrock_client = get_bedrock_client()
    
    # Limit content length to avoid token limits
    truncated_content = content[:25000] if len(content) > 25000 else content
    
    # Create the prompt for generating the summary
    prompt = f"""<source_content>
{truncated_content}
</source_content>

The above content is from the documentation for '{source_id}'. Please provide a concise summary (3-5 sentences) that describes what this library/tool/framework is about. The summary should help understand what the library/tool/framework accomplishes and the purpose.
"""
    
    try:
        # Prepare request body based on model family
        if "claude" in text_model.lower():
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 150,
                "temperature": 0.3,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
        elif "titan" in text_model.lower():
            body = json.dumps({
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 150,
                    "temperature": 0.3
                }
            })
        else:
            # Default to Claude format
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 150,
                "temperature": 0.3,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })

        # Call Bedrock for text generation
        response = bedrock_client.invoke_model(
            modelId=text_model,
            body=body,
            contentType='application/json',
            accept='application/json'
        )
        
        # Parse response based on model
        response_body = json.loads(response['body'].read())
        
        if "claude" in text_model.lower():
            summary = response_body['content'][0]['text'].strip()
        elif "titan" in text_model.lower():
            summary = response_body['results'][0]['outputText'].strip()
        else:
            # Try to extract text from common response formats
            if 'content' in response_body and len(response_body['content']) > 0:
                summary = response_body['content'][0]['text'].strip()
            elif 'results' in response_body and len(response_body['results']) > 0:
                summary = response_body['results'][0]['outputText'].strip()
            else:
                return default_summary
        
        # Ensure the summary is not too long
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
            
        return summary
    
    except Exception as e:
        print(f"Error generating summary with Bedrock for {source_id}: {e}. Using default summary.")
        return default_summary