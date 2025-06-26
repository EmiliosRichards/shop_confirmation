import json
import logging
from typing import Dict, Any, Optional

from google.generativeai import types as genai_types
from src.llm_clients.gemini_client import GeminiClient
from src.core.config import AppConfig
from src.utils.llm_processing_helpers import save_llm_interaction

logger = logging.getLogger(__name__)

def perform_classification(
    scraped_text: str,
    gemini_client: GeminiClient,
    app_config: AppConfig,
    original_url: str,
    llm_context_dir: str,
    llm_requests_dir: str,
    file_identifier_prefix: str,
    classification_profile: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Uses the LLM to perform a generic classification based on a profile.

    Args:
        scraped_text: The text content scraped from the website.
        gemini_client: The client for interacting with the Gemini LLM.
        app_config: The application configuration object.
        original_url: The original URL being analyzed.
        llm_context_dir: Directory to save LLM context.
        llm_requests_dir: Directory to save LLM requests.
        file_identifier_prefix: A prefix for filenames.
        classification_profile: The profile defining the classification task.

    Returns:
        A dictionary with the classification results or None if it fails.
    """
    prompt_path = classification_profile.get("prompt_path")
    if not prompt_path:
        logger.error("Classification profile is missing 'prompt_path'.")
        return None

    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
    except FileNotFoundError:
        logger.error(f"Classification prompt not found at: {prompt_path}")
        return None

    prompt = prompt_template.format(website_text=scraped_text[:app_config.LLM_MAX_INPUT_CHARS_FOR_SUMMARY])
    
    response_obj = None
    try:
        generation_config = genai_types.GenerationConfig(
            temperature=app_config.llm_temperature_default,
            max_output_tokens=app_config.llm_max_tokens,
            response_mime_type="application/json",
        )
        
        response_obj = gemini_client.generate_content_with_retry(
            contents=[genai_types.ContentDict(parts=[{"text": prompt}], role="user")],
            generation_config=generation_config,
            file_identifier_prefix=file_identifier_prefix,
            triggering_input_row_id=original_url,
            triggering_company_name="N/A"
        )
        
        raw_response_text = response_obj.text if response_obj and hasattr(response_obj, 'text') else 'No text in response'
        
        save_llm_interaction(
            llm_context_dir=llm_context_dir,
            llm_requests_dir=llm_requests_dir,
            file_identifier_prefix=file_identifier_prefix,
            prompt_text=prompt,
            raw_response_text=raw_response_text,
            log_prefix="generic_classification"
        )

        if response_obj and response_obj.text:
            cleaned_response = response_obj.text.strip().replace('```json', '').replace('```', '').strip()
            return json.loads(cleaned_response)
        else:
            logger.error(f"No valid text response for classification for URL '{original_url}'.")
            return None

    except json.JSONDecodeError as e:
        raw_response_text_for_log = response_obj.text if response_obj and hasattr(response_obj, 'text') else 'N/A'
        logger.error(f"Failed to parse JSON response for URL '{original_url}'. Error: {e}. Raw response: {raw_response_text_for_log}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during classification for URL '{original_url}': {e}")
        return None