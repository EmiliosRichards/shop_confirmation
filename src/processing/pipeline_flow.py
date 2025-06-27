"""
Core pipeline execution flow for processing company data for classification.
"""
import pandas as pd
import os
import asyncio
import json
import time
from datetime import datetime
import logging
from typing import List, Dict, Set, Optional, Any, Tuple
from collections import Counter

from src.core.config import AppConfig
from src.llm_clients.gemini_client import GeminiClient
from src.scraper import scrape_website
from src.extractors.llm_tasks.generic_classification_task import perform_classification
from src.utils.helpers import log_row_failure, sanitize_filename_component
from src.processing.url_processor import process_input_url

logger = logging.getLogger(__name__)

PipelineOutput = Tuple[
    pd.DataFrame,
    List[Dict[str, Any]],
    Dict[str, int]
]

def _run_classification_flow(
    processed_url: str,
    row_series: pd.Series,
    app_config: AppConfig,
    gemini_client: GeminiClient,
    run_output_dir: str,
    log_identifier: str,
    index: Any,
    company_name_str: str,
    classification_profile: Dict[str, Any],
    scraped_text: Optional[str] = None
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Executes a generic classification flow for a single row.
    If scraped_text is provided, it skips the scraping step.
    """
    results = {}
    output_columns = classification_profile.get("output_columns", {})
    for key in output_columns.values():
        results[key] = ""

    if scraped_text is None:
        # This path is for single-stage modes. It will be removed from the two-stage flow.
        # The scraping logic will be handled directly in the main loop for two-stage.
        # This is a simplification for the refactor.
        logger.error("Scraping should be handled before calling _run_classification_flow in new design.")
        return results, "Error_Scraping_Not_Done"


    logger.info(f"{log_identifier} Using pre-scraped text for classification.")

    llm_file_prefix = sanitize_filename_component(
        f"Row{index}_{company_name_str[:20]}_{classification_profile.get('prompt_path', 'classify')}", max_len=50
    )
    
    classification_result = perform_classification(
        scraped_text=scraped_text,
        gemini_client=gemini_client,
        app_config=app_config,
        original_url=str(row_series.get("GivenURL", "")),
        llm_context_dir=os.path.join(run_output_dir, app_config.llm_context_subdir),
        llm_requests_dir=os.path.join(run_output_dir, "llm_requests"),
        file_identifier_prefix=llm_file_prefix,
        classification_profile=classification_profile
    )

    if classification_result:
        for key, col_name in output_columns.items():
            results[col_name] = classification_result.get(key, "")
        logger.info(f"{log_identifier} Classification successful for profile: {classification_profile.get('prompt_path')}")
    else:
        logger.warning(f"{log_identifier} Classification LLM call failed for profile: {classification_profile.get('prompt_path')}")
        for col_name in output_columns.values():
            results[col_name] = "Error"

    return results, "Success" # Scraper status is handled outside now


def execute_pipeline_flow(
    df: pd.DataFrame,
    app_config: AppConfig,
    gemini_client: GeminiClient,
    run_output_dir: str,
    llm_context_dir: str,
    llm_requests_dir: str,
    run_id: str,
    failure_writer: Any,
    run_metrics: Dict[str, Any]
) -> PipelineOutput:
    """
    Executes the core data processing flow of the pipeline.
    """
    globally_processed_urls: Set[str] = set()
    attrition_data_list: List[Dict[str, Any]] = []
    row_level_failure_counts: Dict[str, int] = Counter()

    pipeline_loop_start_time = time.time()
    rows_processed_count = 0
    rows_failed_count = 0

    active_profile = app_config.INPUT_COLUMN_PROFILES.get(
        app_config.input_file_profile_name,
        app_config.INPUT_COLUMN_PROFILES['default']
    )
    company_name_col_key = active_profile.get('CompanyName', 'CompanyName')
    url_col_key = active_profile.get('GivenURL', 'GivenURL')

    for i, (index, row_series) in enumerate(df.iterrows()):
        rows_processed_count += 1
        row: pd.Series = row_series
        company_name_str: str = str(row.get(company_name_col_key, f"MissingCompanyName_Row_{index}"))
        given_url_original: Optional[str] = row.get(url_col_key)
        given_url_original_str: str = str(given_url_original) if given_url_original else "MissingURL"

        current_row_number_for_log: int = i + 1
        log_identifier = f"[RowID: {index}, Company: {company_name_str}, URL: {given_url_original_str}]"
        logger.info(f"{log_identifier} --- Processing row {current_row_number_for_log}/{len(df)} ---")

        processed_url, url_status = process_input_url(
            given_url_original, app_config.url_probing_tlds, log_identifier
        )
        if url_status == "InvalidURL":
            df.at[index, 'ScrapingStatus'] = 'InvalidURL'
            run_metrics["scraping_stats"]["scraping_failure_invalid_url"] += 1
            log_row_failure(
                failure_writer, index, company_name_str, given_url_original_str,
                "URL_Validation_InvalidOrMissing",
                f"Invalid or missing URL: {processed_url}", datetime.now().isoformat(),
                json.dumps({"original_url": given_url_original_str, "processed_url": processed_url})
            )
            row_level_failure_counts["URL_Validation_InvalidOrMissing"] += 1
            rows_failed_count += 1
            continue

        try:
            assert processed_url is not None

            # Scrape website once for all classification modes
            target_keywords = app_config.CLASSIFICATION_PROFILES.get(app_config.pipeline_mode, {}).get("target_keywords", [])
            if app_config.pipeline_mode == "two_stage_classification":
                # For two-stage, use keywords from both profiles
                exclusion_keywords = app_config.CLASSIFICATION_PROFILES["exclusion_detection"].get("target_keywords", [])
                positive_keywords = app_config.CLASSIFICATION_PROFILES["positive_criteria_detection"].get("target_keywords", [])
                target_keywords = list(set(exclusion_keywords + positive_keywords))

            _, scraper_status, _, collected_summary_text = asyncio.run(
                scrape_website(
                    processed_url, run_output_dir, company_name_str,
                    globally_processed_urls, index, target_keywords=target_keywords
                )
            )
            df.at[index, 'ScrapingStatus'] = scraper_status

            if scraper_status != "Success" or not collected_summary_text:
                logger.warning(f"{log_identifier} Scraping failed or no text collected. Status: {scraper_status}")
                log_row_failure(
                    failure_writer, index, company_name_str, given_url_original_str,
                    f"Scraping_{scraper_status}", f"Scraping status: {scraper_status}",
                    datetime.now().isoformat(), json.dumps({"processed_url": processed_url})
                )
                row_level_failure_counts[f"Scraping_{scraper_status}"] += 1
                rows_failed_count += 1
                continue

            logger.info(f"{log_identifier} Collected {len(collected_summary_text)} characters for classification.")

            if app_config.pipeline_mode == "two_stage_classification":
                logger.info(f"{log_identifier} Pipeline mode: 'two_stage_classification'.")
                
                # Stage 1: Exclusion Detection
                exclusion_profile = app_config.CLASSIFICATION_PROFILES["exclusion_detection"]
                exclusion_result, _ = _run_classification_flow(
                    processed_url=processed_url, row_series=row, app_config=app_config,
                    gemini_client=gemini_client, run_output_dir=run_output_dir,
                    log_identifier=log_identifier, index=index, company_name_str=company_name_str,
                    classification_profile=exclusion_profile, scraped_text=collected_summary_text
                )
                if exclusion_result:
                    for col, val in exclusion_result.items(): df.at[index, col] = val
                
                if exclusion_result.get("is_excluded", "Error").lower() == "yes":
                    logger.info(f"{log_identifier} Company excluded. Skipping positive criteria check.")
                    continue

                # Stage 2: Positive Criteria Detection
                logger.info(f"{log_identifier} Proceeding to positive criteria check.")
                positive_profile = app_config.CLASSIFICATION_PROFILES["positive_criteria_detection"]
                positive_result, _ = _run_classification_flow(
                    processed_url=processed_url, row_series=row, app_config=app_config,
                    gemini_client=gemini_client, run_output_dir=run_output_dir,
                    log_identifier=log_identifier, index=index, company_name_str=company_name_str,
                    classification_profile=positive_profile, scraped_text=collected_summary_text
                )
                if positive_result:
                    for col, val in positive_result.items(): df.at[index, col] = val

            elif app_config.pipeline_mode in app_config.CLASSIFICATION_PROFILES:
                profile_name = app_config.pipeline_mode
                classification_profile = app_config.CLASSIFICATION_PROFILES[profile_name]
                logger.info(f"{log_identifier} Pipeline mode: '{profile_name}'.")

                classification_result, _ = _run_classification_flow(
                    processed_url=processed_url, row_series=row, app_config=app_config,
                    gemini_client=gemini_client, run_output_dir=run_output_dir,
                    log_identifier=log_identifier, index=index, company_name_str=company_name_str,
                    classification_profile=classification_profile, scraped_text=collected_summary_text
                )
                if classification_result:
                    for col, val in classification_result.items(): df.at[index, col] = val
            else:
                logger.error(f"{log_identifier} Unknown pipeline mode '{app_config.pipeline_mode}'.")
                rows_failed_count += 1

        except Exception as e_row_processing:
            logger.error(
                f"{log_identifier} Unhandled error for row {current_row_number_for_log}: {e_row_processing}",
                exc_info=True
            )
            run_metrics["errors_encountered"].append(
                f"Row error for {company_name_str} (URL: {given_url_original_str}): {str(e_row_processing)}"
            )
            log_row_failure(
                failure_writer, index, company_name_str, given_url_original_str,
                "RowProcessing_UnhandledException", "Unhandled exception in main loop",
                datetime.now().isoformat(),
                json.dumps({
                    "exception_type": type(e_row_processing).__name__,
                    "exception_message": str(e_row_processing)
                })
            )
            row_level_failure_counts["RowProcessing_UnhandledException"] += 1
            rows_failed_count += 1

    run_metrics["tasks"]["pipeline_main_loop_duration_seconds"] = time.time() - pipeline_loop_start_time
    run_metrics["data_processing_stats"]["rows_successfully_processed_main_flow"] = \
        rows_processed_count - rows_failed_count
    run_metrics["data_processing_stats"]["rows_failed_main_flow"] = rows_failed_count
    run_metrics["data_processing_stats"]["row_level_failure_summary"] = dict(row_level_failure_counts)
    logger.info(f"Main processing loop complete. Processed {rows_processed_count} rows.")

    return (
        df,
        attrition_data_list,
        dict(row_level_failure_counts)
    )