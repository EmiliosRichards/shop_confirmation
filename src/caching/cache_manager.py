import os
import hashlib
import logging
import json
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class CacheManager:
    """Handles caching for scraped content and LLM results."""

    def __init__(self, cache_base_dir: str = 'cache'):
        self.scraped_content_dir = os.path.join(cache_base_dir, 'scraped_content')
        self.llm_results_dir = os.path.join(cache_base_dir, 'llm_results')
        self._initialize_cache_directories()

    def _initialize_cache_directories(self):
        """Creates the cache directories if they don't exist."""
        os.makedirs(self.scraped_content_dir, exist_ok=True)
        os.makedirs(self.llm_results_dir, exist_ok=True)

    def _get_cache_key(self, for_url: str) -> str:
        """Generates a consistent cache key for a URL."""
        return hashlib.md5(for_url.encode('utf-8')).hexdigest()

    def get_scrape_result(self, url: str) -> Optional[Dict[str, Any]]:
        """Retrieves a scrape result (status and content) from the cache."""
        cache_key = self._get_cache_key(url)
        cache_file = os.path.join(self.scraped_content_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                logger.error(f"Error reading or parsing cache file {cache_file}: {e}")
                return None
        return None

    def set_scrape_result(self, url: str, status: str, content: Optional[str]):
        """Saves a scrape result to the cache."""
        cache_key = self._get_cache_key(url)
        cache_file = os.path.join(self.scraped_content_dir, f"{cache_key}.json")
        data = {"status": status, "content": content or ""}
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def get_llm_result(self, text_hash: str, prompt_name: str) -> Optional[str]:
        """Retrieves an LLM result from the cache."""
        cache_key = f"{prompt_name}_{text_hash}"
        cache_file = os.path.join(self.llm_results_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
        return None

    def set_llm_result(self, text_hash: str, prompt_name: str, result: str):
        """Saves an LLM result to the cache."""
        cache_key = f"{prompt_name}_{text_hash}"
        cache_file = os.path.join(self.llm_results_dir, f"{cache_key}.json")
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(result)