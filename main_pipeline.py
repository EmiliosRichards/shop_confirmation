import pandas as pd
from typing import List, Dict, Optional, Any
import csv
import logging
import os
import time
import argparse

from dotenv import load_dotenv

from src.data_handling.loader import load_and_preprocess_data
from src.llm_clients.gemini_client import GeminiClient
from src.core.logging_config import setup_logging
from src.core.config import AppConfig
from src.caching.cache_manager import CacheManager
from src.utils.helpers import (
    generate_run_id,
    resolve_path,
    initialize_run_metrics,
    setup_output_directories,
    precompute_input_duplicate_stats,
    initialize_dataframe_columns
)
from src.reporting.metrics_manager import write_run_metrics
from src.processing.pipeline_flow import execute_pipeline_flow
from src.reporting.main_report_orchestrator import generate_all_reports
import shutil
import json

# Explicitly load the .env file from the project root to ensure consistency.
# This is the single source of truth for environment-based configuration.
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

logger = logging.getLogger(__name__)

BASE_FILE_PATH_FOR_RESOLVE = __file__

def main() -> None:
    """
    Main entry point for the phone validation pipeline.
    Orchestrates the entire process from data loading to report generation.
    """
    parser = argparse.ArgumentParser(description="Run the classification pipeline with specified configurations.")
    parser.add_argument('-i', '--input-file', type=str, help='Path to the input data file.')
    parser.add_argument('-r', '--range', type=str, help='Row range to process (e.g., "1-100", "50-", "-200").')
    parser.add_argument('--resume-from', type=str, help='Run ID to resume scraped content from.')
    parser.add_argument('-s', '--suffix', type=str, help='Suffix to append to the run ID.')
    parser.add_argument('-m', '--mode', type=str, help='Pipeline mode to run (e.g., "shop_detection").')
    parser.add_argument('-p', '--profile', type=str, help='Input file column profile name.')
    
    args = parser.parse_args()
    app_config = AppConfig(cli_args=args)
    cache_manager = CacheManager(cache_base_dir=app_config.cache_base_dir)

    if app_config.resume_from_run_id:
        logger.info(f"Attempting to resume from run ID: {app_config.resume_from_run_id}")
        source_run_dir = os.path.join(app_config.output_base_dir, app_config.resume_from_run_id, 'scraped_content')
        if os.path.isdir(source_run_dir):
            for item in os.listdir(source_run_dir):
                s = os.path.join(source_run_dir, item)
                d = os.path.join(cache_manager.scraped_content_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            logger.info(f"Successfully copied scraped content from {app_config.resume_from_run_id} to cache.")
        else:
            logger.warning(f"Could not find scraped_content directory for run ID {app_config.resume_from_run_id}. Proceeding without resuming.")

    pipeline_start_time = time.time()
    
    run_id = generate_run_id(suffix=app_config.run_id_suffix)
    run_metrics: Dict[str, Any] = initialize_run_metrics(run_id)

    run_output_dir, llm_context_dir, llm_requests_dir = setup_output_directories(app_config, run_id, BASE_FILE_PATH_FOR_RESOLVE, app_config.pipeline_mode)

    config_save_path = os.path.join(run_output_dir, 'run_config.json')
    with open(config_save_path, 'w') as f:
        json.dump(app_config.to_dict(), f, indent=4)
    logger.info(f"Run configuration saved to {config_save_path}")

    log_file_name = f"pipeline_run_{run_id}.log"
    log_file_path = os.path.join(run_output_dir, log_file_name)
    file_log_level_int = getattr(logging, app_config.log_level.upper(), logging.INFO)
    console_log_level_int = getattr(logging, app_config.console_log_level.upper(), logging.WARNING)
    setup_logging(file_log_level=file_log_level_int, console_log_level=console_log_level_int, log_file_path=log_file_path)
    
    logger.info(f"Logging initialized. Run ID: {run_id}")
    logger.info(f"Base output directory for this run: {run_output_dir}")

    input_file_path_abs = resolve_path(app_config.input_excel_file_path, BASE_FILE_PATH_FOR_RESOLVE)
    logger.info(f"Resolved input file path: {input_file_path_abs}")

    failure_log_csv_path = os.path.join(run_output_dir, f"failed_rows_{run_id}.csv")
    logger.info(f"Row-specific failure log for this run will be: {failure_log_csv_path}")

    logger.info("Starting classification pipeline...")
    if not os.path.exists(input_file_path_abs):
        logger.error(f"CRITICAL: Input file not found at resolved path: {input_file_path_abs}. Exiting.")
        run_metrics["errors_encountered"].append(f"Input file not found: {input_file_path_abs}")
        run_metrics["total_duration_seconds"] = time.time() - pipeline_start_time
        write_run_metrics(metrics=run_metrics, output_dir=run_output_dir if 'run_output_dir' in locals() else ".", run_id=run_id, pipeline_start_time=pipeline_start_time, attrition_data_list_for_metrics=[], canonical_domain_journey_data={})
        return

    gemini_client: Optional[GeminiClient] = None
    try:
        gemini_client = GeminiClient(config=app_config)
        logger.info("GeminiClient initialized successfully.")
    except ValueError as ve:
        logger.error(f"Failed to initialize GeminiClient: {ve}. Check GEMINI_API_KEY. Pipeline cannot proceed with LLM steps.")
        run_metrics["errors_encountered"].append(f"LLM Extractor init failed: {ve}")
        run_metrics["total_duration_seconds"] = time.time() - pipeline_start_time
        write_run_metrics(metrics=run_metrics, output_dir=run_output_dir, run_id=run_id, pipeline_start_time=pipeline_start_time, attrition_data_list_for_metrics=[], canonical_domain_journey_data={})
        return
    except Exception as e:
        logger.error(f"Unexpected error initializing GeminiClient: {e}", exc_info=True)
        run_metrics["errors_encountered"].append(f"LLM Extractor init unexpected error: {e}")
        run_metrics["total_duration_seconds"] = time.time() - pipeline_start_time
        write_run_metrics(metrics=run_metrics, output_dir=run_output_dir, run_id=run_id, pipeline_start_time=pipeline_start_time, attrition_data_list_for_metrics=[], canonical_domain_journey_data={})
        return
    
    if gemini_client is None:
        logger.error("GeminiClient is None after initialization attempt. Exiting.")
        return

    df: Optional[pd.DataFrame] = None
    task_start_time = time.time()
    try:
        logger.info(f"Attempting to load data from: {input_file_path_abs}")
        df = load_and_preprocess_data(input_file_path_abs, app_config_instance=app_config)
        if df is not None:
            logger.info(f"Successfully loaded and preprocessed data. Shape: {df.shape}.")
            run_metrics["data_processing_stats"]["input_rows_count"] = len(df)
        else:
            logger.error(f"Failed to load data from {input_file_path_abs}. DataFrame is None.")
            run_metrics["errors_encountered"].append(f"Data loading failed: DataFrame is None from {input_file_path_abs}")
            run_metrics["tasks"]["load_and_preprocess_data_duration_seconds"] = time.time() - task_start_time
            run_metrics["total_duration_seconds"] = time.time() - pipeline_start_time
            write_run_metrics(metrics=run_metrics, output_dir=run_output_dir, run_id=run_id, pipeline_start_time=pipeline_start_time, attrition_data_list_for_metrics=[], canonical_domain_journey_data={})
            return
    except Exception as e:
        logger.error(f"Error loading data in main: {e}", exc_info=True)
        run_metrics["errors_encountered"].append(f"Data loading exception: {str(e)}")
        run_metrics["tasks"]["load_and_preprocess_data_duration_seconds"] = time.time() - task_start_time
        run_metrics["total_duration_seconds"] = time.time() - pipeline_start_time
        write_run_metrics(metrics=run_metrics, output_dir=run_output_dir, run_id=run_id, pipeline_start_time=pipeline_start_time, attrition_data_list_for_metrics=[], canonical_domain_journey_data={})
        return
    run_metrics["tasks"]["load_and_preprocess_data_duration_seconds"] = time.time() - task_start_time
 
    if df is None:
        logger.error("DataFrame is None after loading attempt, cannot proceed.")
        return
    assert df is not None, "DataFrame loading failed, assertion."

    if df is not None:
        df = initialize_dataframe_columns(df, app_config)

    pre_comp_start_time = time.time()
    if df is not None:
        df = precompute_input_duplicate_stats(df, app_config, run_metrics)
    logger.info(f"Input duplicate pre-computation complete. Duration: {time.time() - pre_comp_start_time:.2f}s")
    run_metrics["tasks"]["pre_computation_duplicate_counts_duration_seconds"] = time.time() - pre_comp_start_time
    
    attrition_data_list: List[Dict[str, Any]] = []
    
    failure_log_file_handle = None
    failure_writer = None
    try:
        failure_log_file_handle = open(failure_log_csv_path, 'w', newline='', encoding='utf-8')
        failure_writer = csv.writer(failure_log_file_handle)
        failure_writer.writerow(['log_timestamp', 'input_row_identifier', 'CompanyName', 'GivenURL', 'stage_of_failure', 'error_reason', 'error_details', 'Associated_Pathful_Canonical_URL'])

        logger.info("Starting core pipeline processing flow...")
        (df, attrition_data_list, row_level_failure_counts
        ) = execute_pipeline_flow(
            df,
            app_config=app_config,
            gemini_client=gemini_client,
            run_output_dir=run_output_dir,
            llm_context_dir=llm_context_dir,
            llm_requests_dir=llm_requests_dir,
            run_id=run_id,
            failure_writer=failure_writer,
            run_metrics=run_metrics,
            cache_manager=cache_manager
        )
        run_metrics["data_processing_stats"]["row_level_failure_summary"] = row_level_failure_counts
        logger.info("Core pipeline processing flow finished.")

        if df is not None:
            generate_all_reports(
                df=df,
                app_config=app_config,
                run_id=run_id,
                run_output_dir=run_output_dir,
                run_metrics=run_metrics
            )

    except Exception as pipeline_exec_error:
        logger.error(f"An unhandled error occurred during pipeline execution or reporting: {pipeline_exec_error}", exc_info=True)
        run_metrics["errors_encountered"].append(f"Pipeline execution/reporting error: {str(pipeline_exec_error)}")
    finally:
        if failure_log_file_handle:
            try:
                failure_log_file_handle.close()
            except Exception as e_close:
                logger.error(f"Error closing failure log CSV: {e_close}")
    
    run_metrics["total_duration_seconds"] = time.time() - pipeline_start_time
    write_run_metrics(
        metrics=run_metrics,
        output_dir=run_output_dir,
        run_id=run_id,
        pipeline_start_time=pipeline_start_time,
        attrition_data_list_for_metrics=attrition_data_list,
        canonical_domain_journey_data={}
    )
    logger.info(f"Pipeline run {run_id} finished. Total duration: {run_metrics['total_duration_seconds']:.2f}s.")
    logger.info(f"Run metrics file: {os.path.join(run_output_dir, f'run_metrics_{run_id}.md')}")
    logger.info(f"All outputs for this run are in: {run_output_dir}")

if __name__ == '__main__':
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()