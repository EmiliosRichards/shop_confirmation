"""
Core pipeline execution flow for processing company data.

This module orchestrates the main data processing pipeline, which involves:
1.  Iterating through input company data (typically from a DataFrame).
2.  Processing and validating input URLs.
3.  Scraping website content for valid URLs.
4.  A 3-stage LLM (Large Language Model) process:
    a.  Generate a summary of the website content.
    b.  Extract detailed company attributes from the summary.
    c.  Generate sales insights by comparing extracted attributes against
        golden partner profiles.
5.  Collecting metrics, logging failures, and preparing outputs.

The pipeline is designed to be resilient, handling errors at each stage and
continuing processing for other rows where possible. It also tracks various
data points for reporting and analysis, such as scraper statuses and LLM
call statistics.
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
from src.core.schemas import (
    WebsiteTextSummary, DetailedCompanyAttributes, GoldenPartnerMatchOutput, ShopClassificationOutput
)
from src.data_handling.consolidator import get_canonical_base_url
from src.llm_clients.gemini_client import GeminiClient
from src.scraper import scrape_website
from src.extractors.llm_tasks.summarize_task import generate_website_summary
from src.extractors.llm_tasks.extract_attributes_task import extract_detailed_attributes
from src.extractors.llm_tasks.generate_insights_task import generate_sales_insights
from src.extractors.llm_tasks.is_shop_task import classify_is_shop
from src.utils.helpers import log_row_failure, sanitize_filename_component
from src.processing.url_processor import process_input_url

logger = logging.getLogger(__name__)

# Define a type alias for the complex return tuple for better readability
PipelineOutput = Tuple[
    pd.DataFrame,
    List[GoldenPartnerMatchOutput],
    List[Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, str],
    Dict[str, List[str]],
    Dict[str, Optional[str]],
    Dict[str, int]
]

def _run_shop_detection_flow(
    processed_url: str,
    row_series: pd.Series,
    app_config: AppConfig,
    gemini_client: GeminiClient,
    run_output_dir: str,
    globally_processed_urls: Set[str],
    log_identifier: str,
    index: Any,
    company_name_str: str,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Executes the shop detection flow for a single row.

    This involves scraping the website with targeted keywords and then using an
    LLM to classify if the site is a shop.

    Args:
        processed_url: The validated URL to process.
        row_series: The pandas Series for the current row.
        app_config: The application configuration.
        gemini_client: The Gemini client for LLM calls.
        run_output_dir: Directory for scraper outputs.
        globally_processed_urls: A set of URLs that have already been processed.
        log_identifier: A string for logging purposes.
        index: The row index.
        company_name_str: The company name.

    Returns:
        A tuple containing:
        - A dictionary with the results ('is_shop', 'is_shop_confidence', 'is_shop_evidence').
        - The scraper status.
    """
    results = {
        "is_shop": None,
        "is_shop_confidence": None,
        "is_shop_evidence": None,
        "scraper_status": "Not_Run",
    }

    logger.info(f"{log_identifier} Starting shop detection scraping for: {processed_url}")
    _, scraper_status, _, collected_summary_text = asyncio.run(
        scrape_website(
            processed_url,
            run_output_dir,
            company_name_str,
            globally_processed_urls,
            index,
            target_keywords=app_config.shop_detection_target_keywords,
        )
    )
    results["scraper_status"] = scraper_status

    if scraper_status != "Success" or not collected_summary_text:
        logger.warning(f"{log_identifier} Scraping failed or no text collected for shop detection. Status: {scraper_status}")
        return results, scraper_status

    logger.info(f"{log_identifier} Collected {len(collected_summary_text)} characters for shop detection.")

    llm_file_prefix = sanitize_filename_component(
        f"Row{index}_{company_name_str[:20]}_shop_detect", max_len=50
    )
    shop_class_obj = classify_is_shop(
        scraped_text=collected_summary_text,
        gemini_client=gemini_client,
        app_config=app_config,
        original_url=str(row_series.get("GivenURL", "")),
        llm_context_dir=os.path.join(run_output_dir, app_config.llm_context_subdir),
        llm_requests_dir=os.path.join(run_output_dir, "llm_requests"),
        file_identifier_prefix=llm_file_prefix
    )

    if shop_class_obj:
        results["is_shop"] = shop_class_obj.get("is_shop")
        results["is_shop_confidence"] = shop_class_obj.get("is_shop_confidence")
        results["is_shop_evidence"] = shop_class_obj.get("is_shop_evidence")
        logger.info(f"{log_identifier} Shop detection successful: Is Shop = {results['is_shop']}")
    else:
        logger.warning(f"{log_identifier} Shop detection LLM call failed.")
        results["is_shop"] = "Error" # Mark as error

    return results, scraper_status


def _run_full_analysis_flow(
    gemini_client: GeminiClient,
    app_config: AppConfig,
    collected_summary_text: str,
    given_url_original_str: str,
    llm_context_dir: str,
    llm_requests_dir: str,
    index: Any,
    company_name_str: str,
    golden_partner_summaries: List[Dict[str, str]],
    run_metrics: Dict[str, Any],
    log_identifier: str,
    failure_writer: Any, # csv.writer object
    row_level_failure_counts: Dict[str, int],
    all_golden_partner_match_outputs: List[GoldenPartnerMatchOutput],
    rows_failed_count: int
) -> Tuple[Optional[WebsiteTextSummary], Optional[DetailedCompanyAttributes], Optional[GoldenPartnerMatchOutput], int]:
    """
    Executes the full 3-stage LLM analysis flow for a single row.

    This involves:
    1. Summarizing scraped text.
    2. Extracting attributes from the summary.
    3. Generating sales insights by comparing attributes to golden partners.

    Args:
        gemini_client: The Gemini client for LLM calls.
        app_config: The application configuration.
        collected_summary_text: The scraped text to analyze.
        given_url_original_str: The original URL for logging.
        llm_context_dir: Directory for LLM context files.
        llm_requests_dir: Directory for LLM request files.
        index: The row index.
        company_name_str: The company name.
        golden_partner_summaries: A list of golden partner summaries.
        run_metrics: A dictionary for tracking metrics.
        log_identifier: A string for logging.
        failure_writer: A CSV writer for logging failures.
        row_level_failure_counts: A counter for failure types.
        all_golden_partner_match_outputs: A list to store final outputs.
        rows_failed_count: The current count of failed rows.

    Returns:
        A tuple containing the summary object, attributes object, final match output,
        and the updated count of failed rows.
    """
    website_summary_obj: Optional[WebsiteTextSummary] = None
    detailed_attributes_obj: Optional[DetailedCompanyAttributes] = None
    final_match_output: Optional[GoldenPartnerMatchOutput] = None

    logger.info(f"{log_identifier} Collected {len(collected_summary_text)} characters for LLM processing.")

    # --- LLM Call 1: Generate Website Summary ---
    llm_file_prefix_row = sanitize_filename_component(
        f"Row{index}_{company_name_str[:20]}_{str(time.time())[-5:]}", max_len=50
    )

    summary_obj_tuple = generate_website_summary(
        gemini_client=gemini_client,
        config=app_config,
        original_url=given_url_original_str,
        scraped_text=collected_summary_text,
        llm_context_dir=llm_context_dir,
        llm_requests_dir=llm_requests_dir,
        file_identifier_prefix=llm_file_prefix_row,
        triggering_input_row_id=index,
        triggering_company_name=company_name_str
    )
    website_summary_obj = summary_obj_tuple[0]
    if summary_obj_tuple[2]:  # token_stats
        run_metrics["llm_processing_stats"]["total_llm_prompt_tokens"] += \
            summary_obj_tuple[2].get("prompt_tokens", 0)
        run_metrics["llm_processing_stats"]["total_llm_completion_tokens"] += \
            summary_obj_tuple[2].get("completion_tokens", 0)
        run_metrics["llm_processing_stats"]["total_llm_tokens_overall"] += \
            summary_obj_tuple[2].get("total_tokens", 0)
        run_metrics["llm_processing_stats"]["llm_calls_summary_generation"] = \
            run_metrics["llm_processing_stats"].get("llm_calls_summary_generation", 0) + 1

    if not website_summary_obj or not website_summary_obj.summary:
        logger.warning(f"{log_identifier} LLM Call 1 (Summarization) failed. Raw: {summary_obj_tuple[1]}")
        log_row_failure(
            failure_writer, index, company_name_str, given_url_original_str,
            "LLM_Summarization_Failed", "Failed to generate website summary.",
            datetime.now().isoformat(),
            json.dumps({"raw_response": summary_obj_tuple[1] or "N/A"})
        )
        row_level_failure_counts["LLM_Summarization_Failed"] += 1
        rows_failed_count += 1
        all_golden_partner_match_outputs.append(
            GoldenPartnerMatchOutput(
                analyzed_company_url=given_url_original_str,
                analyzed_company_attributes=DetailedCompanyAttributes(
                    input_summary_url=given_url_original_str
                ),
                match_rationale_features=["LLM Summarization Failed"]
            )
        )
        return website_summary_obj, detailed_attributes_obj, final_match_output, rows_failed_count
    logger.info(f"{log_identifier} LLM Call 1 (Summarization) successful.")

    # --- LLM Call 2: Extract Detailed Attributes ---
    attributes_obj_tuple = extract_detailed_attributes(
        gemini_client=gemini_client,
        config=app_config,
        summary_obj=website_summary_obj,
        llm_context_dir=llm_context_dir,
        llm_requests_dir=llm_requests_dir,
        file_identifier_prefix=llm_file_prefix_row,
        triggering_input_row_id=index,
        triggering_company_name=company_name_str
    )
    detailed_attributes_obj = attributes_obj_tuple[0]
    if attributes_obj_tuple[2]:  # token_stats
        run_metrics["llm_processing_stats"]["total_llm_prompt_tokens"] += \
            attributes_obj_tuple[2].get("prompt_tokens", 0)
        run_metrics["llm_processing_stats"]["total_llm_completion_tokens"] += \
            attributes_obj_tuple[2].get("completion_tokens", 0)
        run_metrics["llm_processing_stats"]["total_llm_tokens_overall"] += \
            attributes_obj_tuple[2].get("total_tokens", 0)
        run_metrics["llm_processing_stats"]["llm_calls_attribute_extraction"] = \
            run_metrics["llm_processing_stats"].get("llm_calls_attribute_extraction", 0) + 1

    if not detailed_attributes_obj:
        logger.warning(f"{log_identifier} LLM Call 2 (Attribute Extraction) failed. Raw: {attributes_obj_tuple[1]}")
        log_row_failure(
            failure_writer, index, company_name_str, given_url_original_str,
            "LLM_AttributeExtraction_Failed", "Failed to extract detailed attributes.",
            datetime.now().isoformat(),
            json.dumps({"raw_response": attributes_obj_tuple[1] or "N/A"})
        )
        row_level_failure_counts["LLM_AttributeExtraction_Failed"] += 1
        rows_failed_count += 1
        all_golden_partner_match_outputs.append(
            GoldenPartnerMatchOutput(
                analyzed_company_url=website_summary_obj.original_url
                if website_summary_obj else given_url_original_str,
                analyzed_company_attributes=DetailedCompanyAttributes(
                    input_summary_url=website_summary_obj.original_url
                    if website_summary_obj else given_url_original_str
                ),
                match_rationale_features=["LLM Attribute Extraction Failed"]
            )
        )
        return website_summary_obj, detailed_attributes_obj, final_match_output, rows_failed_count
    logger.info(f"{log_identifier} LLM Call 2 (Attribute Extraction) successful.")

    # --- LLM Call 3: Generate Sales Insights & Compare ---
    sales_insights_obj_tuple = generate_sales_insights(
        gemini_client=gemini_client,
        config=app_config,
        target_attributes=detailed_attributes_obj,
        website_summary_obj=website_summary_obj,
        golden_partner_summaries=golden_partner_summaries,
        llm_context_dir=llm_context_dir,
        llm_requests_dir=llm_requests_dir,
        file_identifier_prefix=llm_file_prefix_row,
        triggering_input_row_id=index,
        triggering_company_name=company_name_str
    )
    final_match_output = sales_insights_obj_tuple[0]
    if sales_insights_obj_tuple[2]:  # token_stats
        run_metrics["llm_processing_stats"]["total_llm_prompt_tokens"] += \
            sales_insights_obj_tuple[2].get("prompt_tokens", 0)
        run_metrics["llm_processing_stats"]["total_llm_completion_tokens"] += \
            sales_insights_obj_tuple[2].get("completion_tokens", 0)
        run_metrics["llm_processing_stats"]["total_llm_tokens_overall"] += \
            sales_insights_obj_tuple[2].get("total_tokens", 0)
        run_metrics["llm_processing_stats"]["llm_calls_sales_insights"] = \
            run_metrics["llm_processing_stats"].get("llm_calls_sales_insights", 0) + 1

    if not final_match_output:
        logger.warning(f"{log_identifier} LLM Call 3 (Sales Insights) failed. Raw: {sales_insights_obj_tuple[1]}")
        log_row_failure(
            failure_writer, index, company_name_str, given_url_original_str,
            "LLM_SalesInsights_Failed", "Failed to generate sales insights.",
            datetime.now().isoformat(),
            json.dumps({"raw_response": sales_insights_obj_tuple[1] or "N/A"})
        )
        row_level_failure_counts["LLM_SalesInsights_Failed"] += 1
        # Still add a partial output if attributes were extracted
        all_golden_partner_match_outputs.append(
            GoldenPartnerMatchOutput(
                analyzed_company_url=detailed_attributes_obj.input_summary_url,
                analyzed_company_attributes=detailed_attributes_obj,
                match_rationale_features=["LLM Sales Insights Generation Failed"]
            )
        )
    else:
        logger.info(f"{log_identifier} LLM Call 3 (Sales Insights) successful.")
        all_golden_partner_match_outputs.append(final_match_output)

    return website_summary_obj, detailed_attributes_obj, final_match_output, rows_failed_count


def execute_pipeline_flow(
    df: pd.DataFrame,
    app_config: AppConfig,
    gemini_client: GeminiClient,
    run_output_dir: str,
    llm_context_dir: str,
    llm_requests_dir: str,
    run_id: str,
    failure_writer: Any,  # csv.writer object
    run_metrics: Dict[str, Any],
    golden_partner_summaries: List[Dict[str, str]]
) -> PipelineOutput:
    """
    Executes the core data processing flow of the pipeline.

    The flow per input row is:
    1. URL Validation: Input URL is processed and validated.
    2. Scrape Website: Content is scraped from the validated URL.
    3. LLM - Summarize: Scraped text is summarized by an LLM.
    4. LLM - Extract Attributes: Detailed attributes are extracted from the summary by an LLM.
    5. LLM - Compare & Sales Line: Attributes are compared to golden partner profiles,
       and sales insights are generated by an LLM.

    Failures at any step are logged, and the pipeline attempts to continue with the next row.

    Args:
        df: Input DataFrame containing company data.
        app_config: Application configuration object.
        gemini_client: Client for interacting with the Gemini LLM.
        run_output_dir: Directory where scraper outputs (like HTML files) are stored.
        llm_context_dir: Directory where LLM interaction context (prompts, responses)
                         is stored for debugging and analysis.
        llm_requests_dir: Directory where LLM request payloads are stored.
        run_id: Unique identifier for the current pipeline run.
        failure_writer: A CSV writer object for logging row-level failures.
        run_metrics: A dictionary that will be updated with various processing metrics.
        golden_partner_summaries: A list of dictionaries, where each dictionary
                                  contains the name and summary of a "golden partner."

    Returns:
        A tuple containing:
        - df (pd.DataFrame): The input DataFrame, potentially updated with statuses.
        - all_golden_partner_match_outputs (List[GoldenPartnerMatchOutput]):
          The primary output; a list of results from the final LLM comparison stage.
        - attrition_data_list (List[Dict[str, Any]]): A list of dictionaries,
          each representing a logged failure.
        - canonical_domain_journey_data (Dict[str, Dict[str, Any]]): Data tracking
          processing attempts and outcomes per canonical domain.
        - true_base_scraper_status (Dict[str, str]): Maps canonical true base URLs
          to their overall scraper status.
        - true_base_to_pathful_map (Dict[str, List[str]]): Maps canonical true base
          URLs to a list of actual pathful URLs scraped under them.
        - input_to_canonical_map (Dict[str, Optional[str]]): Maps original input URLs
          to their determined canonical true base URL.
        - row_level_failure_counts (Dict[str, int]): A summary count of failures
          by type.
    """
    globally_processed_urls: Set[str] = set()  # Tracks URLs to avoid re-scraping
    # Stores scraper status for each specific pathful URL attempted
    canonical_site_pathful_scraper_status: Dict[str, str] = {}
    # Maps original input URL to its determined canonical true base domain
    input_to_canonical_map: Dict[str, Optional[str]] = {}

    all_golden_partner_match_outputs: List[GoldenPartnerMatchOutput] = []
    attrition_data_list: List[Dict[str, Any]] = [] # For detailed failure logging
    row_level_failure_counts: Dict[str, int] = Counter()

    # Data structures for Canonical Domain Journey Report
    # Tracks processing details per unique canonical domain encountered
    canonical_domain_journey_data: Dict[str, Dict[str, Any]] = {}
    true_base_scraper_status: Dict[str, str] = {}
    true_base_to_pathful_map: Dict[str, List[str]] = {}

    pipeline_loop_start_time = time.time()
    rows_processed_count = 0
    rows_failed_count = 0

    # Pre-fetch company name and URL column names from AppConfig
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

        current_row_number_for_log: int = i + 1  # 1-based for logging
        log_identifier = f"[RowID: {index}, Company: {company_name_str}, URL: {given_url_original_str}]"
        logger.info(f"{log_identifier} --- Processing row {current_row_number_for_log}/{len(df)} ---")

        current_row_scraper_status: str = "Not_Run"
        final_canonical_entry_url: Optional[str] = None  # Pathful canonical URL from scraper
        true_base_domain_for_row: Optional[str] = None  # True base domain

        website_summary_obj: Optional[WebsiteTextSummary] = None
        detailed_attributes_obj: Optional[DetailedCompanyAttributes] = None
        final_match_output: Optional[GoldenPartnerMatchOutput] = None

        # --- 1. URL Processing ---
        processed_url, url_status = process_input_url(
            given_url_original, app_config.url_probing_tlds, log_identifier
        )
        if url_status == "InvalidURL":
            df.at[index, 'ScrapingStatus'] = 'InvalidURL'
            current_row_scraper_status = 'InvalidURL'
            run_metrics["scraping_stats"]["scraping_failure_invalid_url"] += 1
            log_row_failure(
                failure_writer, index, company_name_str, given_url_original_str,
                "URL_Validation_InvalidOrMissing",
                f"Invalid or missing URL: {processed_url}", datetime.now().isoformat(),
                json.dumps({"original_url": given_url_original_str, "processed_url": processed_url})
            )
            row_level_failure_counts["URL_Validation_InvalidOrMissing"] += 1
            rows_failed_count += 1
            all_golden_partner_match_outputs.append(
                GoldenPartnerMatchOutput(
                    analyzed_company_url=given_url_original_str,
                    analyzed_company_attributes=DetailedCompanyAttributes(
                        input_summary_url=given_url_original_str
                    ),
                    match_rationale_features=[f"Failed at URL Validation: {url_status}"]
                )
            )
            continue
        # --- End URL Processing ---

        try:
            assert processed_url is not None

            # --- DISPATCH BASED ON PIPELINE MODE ---
            if app_config.pipeline_mode == 'shop_detection':
                logger.info(f"{log_identifier} Pipeline mode: 'shop_detection'. Running shop detection flow.")
                
                shop_detection_result, scraper_status_optional = _run_shop_detection_flow(
                    processed_url=processed_url,
                    row_series=row,
                    app_config=app_config,
                    gemini_client=gemini_client,
                    run_output_dir=run_output_dir,
                    globally_processed_urls=globally_processed_urls,
                    log_identifier=log_identifier,
                    index=index,
                    company_name_str=company_name_str,
                )

                scraper_status = scraper_status_optional or "Detection_Flow_Error"
                df.at[index, 'ScrapingStatus'] = scraper_status
                current_row_scraper_status = scraper_status

                if shop_detection_result:
                    df.at[index, 'is_shop'] = shop_detection_result.get('is_shop')
                    df.at[index, 'is_shop_confidence'] = shop_detection_result.get('is_shop_confidence')
                    df.at[index, 'is_shop_evidence'] = shop_detection_result.get('is_shop_evidence')
                else:
                    df.at[index, 'is_shop'] = None
                    df.at[index, 'is_shop_confidence'] = None
                    df.at[index, 'is_shop_evidence'] = "LLM call failed"

                if current_row_scraper_status != "Success":
                    logger.warning(f"{log_identifier} Shop detection flow failed with scraper status: {current_row_scraper_status}")
                    log_row_failure(
                        failure_writer, index, company_name_str, given_url_original_str,
                        f"Scraping_{current_row_scraper_status}",
                        f"Shop detection scraper status: {current_row_scraper_status}", datetime.now().isoformat(),
                        json.dumps({"processed_url": processed_url})
                    )
                    row_level_failure_counts[f"Scraping_{current_row_scraper_status}"] += 1
                    rows_failed_count += 1
                
                logger.info(f"{log_identifier} Row {current_row_number_for_log} processing complete (shop detection).")

            else:  # 'full_analysis' or default
                if app_config.pipeline_mode != 'full_analysis':
                    logger.warning(f"{log_identifier} Unknown pipeline mode '{app_config.pipeline_mode}'. Defaulting to 'full_analysis'.")
                
                logger.info(f"{log_identifier} Pipeline mode: 'full_analysis'. Running full analysis flow.")
                
                run_metrics["scraping_stats"]["urls_processed_for_scraping"] += 1
                scrape_task_start_time = time.time()

                # --- 2. Scrape Website ---
                logger.info(f"{log_identifier} Starting website scraping for: {processed_url}")
                _, scraper_status, final_canonical_entry_url, collected_summary_text = asyncio.run(
                    scrape_website(processed_url, run_output_dir, company_name_str, globally_processed_urls, index)
                )
                run_metrics["tasks"].setdefault("scrape_website_total_duration_seconds", 0)
                run_metrics["tasks"]["scrape_website_total_duration_seconds"] += (time.time() - scrape_task_start_time)

                df.at[index, 'ScrapingStatus'] = scraper_status
                true_base_domain_for_row = get_canonical_base_url(final_canonical_entry_url) \
                    if final_canonical_entry_url else None
                df.at[index, 'CanonicalEntryURL'] = true_base_domain_for_row
                current_row_scraper_status = scraper_status
                canonical_site_pathful_scraper_status[
                    final_canonical_entry_url if final_canonical_entry_url else processed_url
                ] = scraper_status

                if current_row_scraper_status != "Success":
                    logger.warning(f"{log_identifier} Scraping failed or was skipped. Status: {current_row_scraper_status}")
                    log_row_failure(
                        failure_writer, index, company_name_str, given_url_original_str,
                        f"Scraping_{current_row_scraper_status}",
                        f"Scraper status: {current_row_scraper_status}", datetime.now().isoformat(),
                        json.dumps({
                            "pathful_canonical_url": final_canonical_entry_url,
                            "true_base_domain": true_base_domain_for_row
                        }),
                        associated_pathful_canonical_url=final_canonical_entry_url
                    )
                    row_level_failure_counts[f"Scraping_{current_row_scraper_status}"] += 1
                    rows_failed_count += 1
                    all_golden_partner_match_outputs.append(
                        GoldenPartnerMatchOutput(
                            analyzed_company_url=given_url_original_str,
                            analyzed_company_attributes=DetailedCompanyAttributes(
                                input_summary_url=given_url_original_str
                            ),
                            match_rationale_features=[f"Failed at Scraping: {current_row_scraper_status}"]
                        )
                    )
                    continue

                if not collected_summary_text or not collected_summary_text.strip():
                    logger.warning(f"{log_identifier} No text collected from scraped pages. Skipping LLM calls.")
                    log_row_failure(
                        failure_writer, index, company_name_str, given_url_original_str,
                        "LLM_Input_NoScrapedText", "No text content available after scraping.",
                        datetime.now().isoformat(),
                        json.dumps({"pathful_canonical_url": final_canonical_entry_url})
                    )
                    row_level_failure_counts["LLM_Input_NoScrapedText"] += 1
                    rows_failed_count += 1
                    all_golden_partner_match_outputs.append(
                        GoldenPartnerMatchOutput(
                            analyzed_company_url=given_url_original_str,
                            analyzed_company_attributes=DetailedCompanyAttributes(
                                input_summary_url=given_url_original_str
                            ),
                            match_rationale_features=["No text collected from website scraping"]
                        )
                    )
                    continue
                
                (
                    website_summary_obj,
                    detailed_attributes_obj,
                    final_match_output,
                    rows_failed_count,
                ) = _run_full_analysis_flow(
                    gemini_client=gemini_client,
                    app_config=app_config,
                    collected_summary_text=collected_summary_text,
                    given_url_original_str=given_url_original_str,
                    llm_context_dir=llm_context_dir,
                    llm_requests_dir=llm_requests_dir,
                    index=index,
                    company_name_str=company_name_str,
                    golden_partner_summaries=golden_partner_summaries,
                    run_metrics=run_metrics,
                    log_identifier=log_identifier,
                    failure_writer=failure_writer,
                    row_level_failure_counts=row_level_failure_counts,
                    all_golden_partner_match_outputs=all_golden_partner_match_outputs,
                    rows_failed_count=rows_failed_count,
                )

                if not website_summary_obj or not detailed_attributes_obj:
                    continue

                input_to_canonical_map[given_url_original_str] = true_base_domain_for_row

                if true_base_domain_for_row:
                    if true_base_domain_for_row not in canonical_domain_journey_data:
                        canonical_domain_journey_data[true_base_domain_for_row] = {
                            "Input_Row_IDs": set(), "Input_CompanyNames": set(),
                            "Input_GivenURLs": set(), "Pathful_URLs_Attempted_List": set(),
                            "Overall_Scraper_Status_For_Domain": "Unknown",
                            "LLM_Stages_Attempted": 0, "LLM_Stages_Succeeded": 0
                        }
                    journey_entry = canonical_domain_journey_data[true_base_domain_for_row]
                    journey_entry["Input_Row_IDs"].add(index)
                    journey_entry["Input_CompanyNames"].add(company_name_str)
                    journey_entry["Input_GivenURLs"].add(given_url_original_str)
                    if final_canonical_entry_url:
                        journey_entry["Pathful_URLs_Attempted_List"].add(final_canonical_entry_url)
                    journey_entry["Overall_Scraper_Status_For_Domain"] = current_row_scraper_status
                    
                    journey_entry["LLM_Stages_Attempted"] = 3
                    current_succeeded_stages = 0
                    if website_summary_obj: current_succeeded_stages += 1
                    if detailed_attributes_obj: current_succeeded_stages += 1
                    if final_match_output: current_succeeded_stages += 1
                    journey_entry["LLM_Stages_Succeeded"] = max(
                        journey_entry.get("LLM_Stages_Succeeded", 0), current_succeeded_stages
                    )

                logger.info(f"{log_identifier} Row {current_row_number_for_log} processing complete.")

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
                }),
                associated_pathful_canonical_url=final_canonical_entry_url
            )
            row_level_failure_counts["RowProcessing_UnhandledException"] += 1
            rows_failed_count += 1
            all_golden_partner_match_outputs.append(
                GoldenPartnerMatchOutput(
                    analyzed_company_url=given_url_original_str,
                    analyzed_company_attributes=DetailedCompanyAttributes(
                        input_summary_url=given_url_original_str
                    ),
                    match_rationale_features=[f"Unhandled Exception: {str(e_row_processing)}"]
                )
            )

    run_metrics["tasks"]["pipeline_main_loop_duration_seconds"] = time.time() - pipeline_loop_start_time
    run_metrics["data_processing_stats"]["rows_successfully_processed_main_flow"] = \
        rows_processed_count - rows_failed_count
    run_metrics["data_processing_stats"]["rows_failed_main_flow"] = rows_failed_count
    run_metrics["data_processing_stats"]["row_level_failure_summary"] = dict(row_level_failure_counts)
    logger.info(f"Main processing loop complete. Processed {rows_processed_count} rows.")


    # Populate true_base_scraper_status and true_base_to_pathful_map
    # This aggregates status from individual pathful URL scrapes to their true base domain.
    for pathful_url, status in canonical_site_pathful_scraper_status.items():
        true_base = get_canonical_base_url(pathful_url)
        if true_base:
            true_base_to_pathful_map.setdefault(true_base, []).append(pathful_url)
            # Prioritize "Success" status for a domain if any of its pages succeeded.
            # Otherwise, take the status of one of its pages (could be an error state).
            if true_base not in true_base_scraper_status or status == "Success":
                true_base_scraper_status[true_base] = status
            elif true_base_scraper_status[true_base] != "Success" and "Error" not in status:
                # Prefer a non-error status if current is an error and new one isn't
                true_base_scraper_status[true_base] = status
            # If current is non-success, non-error, and new is error, keep current.
            # If both are errors, the last one processed for that true_base will stick.

    logger.info("Pipeline flow execution finished.")
    
    return (
        df,
        all_golden_partner_match_outputs,
        attrition_data_list,
        canonical_domain_journey_data,
        true_base_scraper_status,
        true_base_to_pathful_map,
        input_to_canonical_map,
        dict(row_level_failure_counts)
    )