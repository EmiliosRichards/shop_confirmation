import json
import logging
from typing import Dict, Any, Optional

from google.generativeai.types import GenerationConfig, ContentDict
from src.llm_clients.gemini_client import GeminiClient
from src.core.config import AppConfig
from src.utils.llm_processing_helpers import save_llm_interaction

logger = logging.getLogger(__name__)

def classify_is_shop(
    scraped_text: str,
    gemini_client: GeminiClient,
    app_config: AppConfig,
    original_url: str,
    llm_context_dir: str,
    llm_requests_dir: str,
    file_identifier_prefix: str
) -> Optional[Dict[str, Any]]:
    """
    Uses the LLM to classify if a website is a shop based on its scraped text.

    Args:
        scraped_text: The text content scraped from the website.
        gemini_client: The client for interacting with the Gemini LLM.
        app_config: The application configuration object.
        original_url: The original URL being analyzed, for logging purposes.

    Returns:
        A dictionary containing the classification results (is_shop, confidence_score, evidence)
        or None if the process fails.
    """
    try:
        with open(app_config.prompt_path_shop_detection, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
    except FileNotFoundError:
        logger.error(f"Shop detection prompt not found at: {app_config.prompt_path_shop_detection}")
        return None

    prompt = prompt_template.format(website_text=scraped_text[:app_config.LLM_MAX_INPUT_CHARS_FOR_SUMMARY])
    
    response_obj = None  # Initialize response_obj to None
    try:
        generation_config = GenerationConfig(
            temperature=app_config.llm_temperature_default,
            max_output_tokens=app_config.llm_max_tokens,
            response_mime_type="application/json",
        )
        
        response_obj = gemini_client.generate_content_with_retry(
            contents=[ContentDict(parts=[{"text": prompt}], role="user")],
            generation_config=generation_config,
            file_identifier_prefix=file_identifier_prefix,
            triggering_input_row_id=original_url, # Using URL as identifier
            triggering_company_name="N/A" # Company name not available here
        )
        
        raw_response_text = response_obj.text if response_obj and hasattr(response_obj, 'text') else 'No text in response'
        
        save_llm_interaction(
            llm_context_dir=llm_context_dir,
            llm_requests_dir=llm_requests_dir,
            file_identifier_prefix=file_identifier_prefix,
            prompt_text=prompt,
            raw_response_text=raw_response_text,
            log_prefix="is_shop_classification"
        )

        if response_obj and response_obj.text:
            cleaned_response = response_obj.text.strip().replace('```json', '').replace('```', '').strip()
            parsed_response = json.loads(cleaned_response)
            return {
                'is_shop': parsed_response.get('is_shop'),
                'is_shop_confidence': parsed_response.get('confidence_score'),
                'is_shop_evidence': parsed_response.get('evidence', 'No evidence provided.')
            }
        else:
            logger.error(f"No valid text response for shop classification for URL '{original_url}'.")
            return None

    except json.JSONDecodeError as e:
        raw_response_text_for_log = response_obj.text if response_obj and hasattr(response_obj, 'text') else 'N/A'
        logger.error(f"Failed to parse JSON response for URL '{original_url}'. Error: {e}. Raw response: {raw_response_text_for_log}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during shop classification for URL '{original_url}': {e}")
        return None