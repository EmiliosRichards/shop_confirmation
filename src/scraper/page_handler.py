import logging
from typing import Optional, Tuple, Any
from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError

from ..core.config import AppConfig

config_instance = AppConfig()
logger = logging.getLogger(__name__)

async def fetch_page_content(page: Page, url: str, input_row_id: Any, company_name_or_id: str) -> Tuple[Optional[str], Optional[int]]:
    logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Navigating to URL: {url}")
    try:
        response = await page.goto(url, timeout=config_instance.default_navigation_timeout, wait_until='domcontentloaded')
        if response:
            logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Navigation to {url} successful. Status: {response.status}")
            if response.ok:
                if config_instance.scraper_networkidle_timeout_ms > 0:
                    logger.debug(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Waiting for networkidle on {url} (timeout: {config_instance.scraper_networkidle_timeout_ms}ms)...")
                    try:
                        await page.wait_for_load_state('networkidle', timeout=config_instance.scraper_networkidle_timeout_ms)
                        logger.debug(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Networkidle achieved for {url}.")
                    except PlaywrightTimeoutError:
                        logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Timeout waiting for networkidle on {url} after {config_instance.scraper_networkidle_timeout_ms}ms. Proceeding with current DOM content.")
                content = await page.content()
                logger.debug(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Content fetched successfully for {url}.")
                return content, response.status
            else:
                logger.warning(f"[RowID: {input_row_id}, Company: {company_name_or_id}] HTTP error for {url}: Status {response.status} {response.status_text}. No content fetched.")
                return None, response.status
        else:
            logger.error(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Failed to get a response object for {url}. Navigation might have failed silently.")
            return None, None
    except PlaywrightTimeoutError:
        logger.error(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Playwright navigation timeout for {url} after {config_instance.default_navigation_timeout / 1000}s.")
        return None, -1 # Specific code for timeout
    except PlaywrightError as e:
        error_message = str(e)
        logger.error(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Playwright error during navigation to {url}: {error_message}")
        if "net::ERR_NAME_NOT_RESOLVED" in error_message: return None, -2 # DNS error
        elif "net::ERR_CONNECTION_REFUSED" in error_message: return None, -3 # Connection refused
        elif "net::ERR_ABORTED" in error_message: return None, -6 # Request aborted, often due to navigation elsewhere
        return None, -4 # Other Playwright error
    except Exception as e:
        logger.error(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Unexpected error fetching page {url}: {type(e).__name__} - {e}", exc_info=True)
        return None, -5 # Generic exception