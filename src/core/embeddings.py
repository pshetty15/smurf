"""
Embedding utilities - extracted from utils.py
Handles Amazon Bedrock and OpenAI embeddings and contextual enrichment
"""
import os
import boto3
import json
import time
import openai
import numpy as np
from typing import List, Tuple
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Client Initialization ---

def get_embedding_provider():
    """Gets the configured embedding provider."""
    return os.getenv("EMBEDDING_PROVIDER", "bedrock").lower()

def get_bedrock_client():
    """Initialize and return Bedrock client with proper configuration."""
    provider = get_embedding_provider()
    
    # Only require Bedrock-specific variables if it's the selected provider
    if provider == "bedrock":
        endpoint_url = os.getenv("BEDROCK_ENDPOINT_URL")
        aws_region = os.getenv("AWS_REGION_NAME", "us-west-2")
        aws_profile = os.getenv("AWS_PROFILE")

        if not endpoint_url:
            raise ValueError("BEDROCK_ENDPOINT_URL environment variable is required for Bedrock provider")
        if not aws_profile:
            raise ValueError("AWS_PROFILE environment variable is required for Bedrock provider")
            
        session = boto3.Session(profile_name=aws_profile)
        return session.client('bedrock-runtime', aws_region, endpoint_url=endpoint_url)
    return None

def get_openai_client():
    """Initialize and return OpenAI client."""
    provider = get_embedding_provider()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider")
        return openai.OpenAI(api_key=api_key)
    return None

# --- Unified Embedding Generation ---

def create_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for a list of texts using the configured provider.
    """
    provider = get_embedding_provider()
    
    if provider == "openai":
        return create_embeddings_openai(texts)
    elif provider == "bedrock":
        return create_embeddings_bedrock(texts)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}. Use 'openai' or 'bedrock'.")

def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using the configured provider.
    """
    if not text:
        return []
    embeddings = create_embeddings([text])
    return embeddings[0] if embeddings else []

# --- Provider-Specific Embedding Implementations ---

def create_embeddings_bedrock(texts: List[str]) -> List[List[float]]:
    """Create embeddings for multiple texts using Amazon Bedrock."""
    if not texts:
        return []
        
    embedding_model = os.getenv("BEDROCK_EMBEDDING_MODEL")
    if not embedding_model:
        raise ValueError("BEDROCK_EMBEDDING_MODEL environment variable is required for Bedrock provider")

    bedrock_client = get_bedrock_client()
    if not bedrock_client:
        print("Bedrock client not initialized.")
        return [[] for _ in texts]

    embeddings = []
    for i, text in enumerate(texts):
        try:
            body = json.dumps({"inputText": text})
            response = bedrock_client.invoke_model(
                body=body,
                modelId=embedding_model,
                accept="application/json",
                contentType="application/json"
            )
            response_body = json.loads(response.get("body").read())
            embeddings.append(response_body.get("embedding"))
        except (ClientError, Exception) as e:
            print(f"AWS Bedrock error for text {i}: {e}")
            embeddings.append([])

    successful_count = sum(1 for emb in embeddings if emb)
    print(f"Successfully created {successful_count}/{len(texts)} embeddings via Bedrock")
    return embeddings

def create_embeddings_openai(texts: List[str]) -> List[List[float]]:
    """Create embeddings for multiple texts using OpenAI."""
    if not texts:
        return []

    openai_client = get_openai_client()
    if not openai_client:
        print("OpenAI client not initialized.")
        return [[] for _ in texts]

    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    try:
        response = openai_client.embeddings.create(input=texts, model=embedding_model)
        embeddings = [item.embedding for item in response.data]
        print(f"Successfully created {len(embeddings)} embeddings via OpenAI")
        return embeddings
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return [[] for _ in texts]


# --- Unified Text Generation ---

def generate_text(prompt: str, max_tokens: int = 2048) -> str:
    """Generate text using the configured provider."""
    # NOTE: We are using the embedding provider for now.
    # A separate TEXT_PROVIDER could be added for more flexibility.
    provider = get_embedding_provider()

    if provider == "openai":
        return generate_text_openai(prompt, max_tokens)
    elif provider == "bedrock":
        return generate_text_bedrock(prompt, max_tokens)
    else:
        raise ValueError(f"Unsupported text generation provider: {provider}")

# --- Provider-Specific Text Generation ---

def generate_text_bedrock(prompt: str, max_tokens: int = 2048) -> str:
    """Uses Amazon Bedrock for text generation."""
    text_model = os.getenv("BEDROCK_TEXT_MODEL")
    if not text_model:
        raise ValueError("BEDROCK_TEXT_MODEL is required for Bedrock provider")
        
    bedrock_client = get_bedrock_client()
    if not bedrock_client:
        print("Bedrock client not initialized.")
        return ""
        
    try:
        # Dynamic payload based on model provider (e.g., Anthropic, Cohere)
        if "anthropic" in text_model:
            payload = {
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": max_tokens,
                "temperature": 0.1,
                "top_p": 0.9,
                "anthropic_version": "bedrock-2023-05-31"
            }
        else: # Default or other models like Cohere
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.1,
            }

        response = bedrock_client.invoke_model(
            body=json.dumps(payload),
            modelId=text_model,
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response['body'].read())

        if "anthropic" in text_model:
            return response_body.get('completion', '').strip()
        elif "cohere" in text_model:
            return response_body['generations'][0]['text'].strip()
        else: # Fallback for other models
            return response_body.get('results', [{}])[0].get('outputText', '').strip()

    except (ClientError, Exception) as e:
        print(f"Error generating text with Bedrock: {e}")
        return ""


def generate_text_openai(prompt: str, max_tokens: int = 2048) -> str:
    """Uses OpenAI for text generation."""
    openai_client = get_openai_client()
    if not openai_client:
        print("OpenAI client not initialized.")
        return ""
        
    try:
        model = os.getenv("OPENAI_MODEL_CHOICE", "gpt-4o-mini")
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating text with OpenAI: {e}")
        return ""

# --- Helper Functions using Text Generation ---

def get_contextual_embedding(chunk: str) -> str:
    """Enhances a text chunk with a generated contextual summary."""
    if not os.getenv("USE_CONTEXTUAL_EMBEDDINGS") == "true":
        return chunk
    
    try:
        prompt = f"""
        Human: Analyze the following text chunk and provide a concise, one-sentence summary of its core topic.
        This summary will be used to improve search recall for semantic search.
        Focus on the main subject and key actions or concepts.

        Text chunk:
        "{chunk}"

        Assistant:
        """
        context = generate_text(prompt)
        return f"{chunk}\n\nContext: {context}"
    except Exception as e:
        print(f"Error generating contextual embedding: {e}. Using original chunk instead.")
        return chunk

def summarize_code_example(code_snippet: str, file_path: str) -> str:
    """Generates a summary for a code snippet."""
    prompt = f"""
    Human: You are an expert code analyst. Summarize the following code snippet from the file '{file_path}'.
    Focus on its primary function, inputs, and outputs in a concise, one-sentence explanation.

    Code snippet:
    ```
    {code_snippet}
    ```

    Assistant:
    """
    try:
        return generate_text(prompt)
    except Exception as e:
        print(f"Error generating code example summary: {e}")
        return "Could not generate summary."

def summarize_source(source_id: str, content_sample: str) -> str:
    """Extract a summary for a source from its content."""
    prompt = f"""
    Human: Read the following content sample from the source '{source_id}'.
    Generate a concise, one-sentence summary of the entire document's purpose.

    Content sample:
    \"\"\"
    {content_sample[:4000]}
    \"\"\"

    Assistant:
    """
    try:
        return generate_text(prompt)
    except Exception as e:
        print(f"Error generating summary for {source_id}: {e}. Using default summary.")
        return "No summary available."