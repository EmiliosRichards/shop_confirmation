"""
Orchestrates the generation of classification reports.
"""
import pandas as pd
import os
import logging
import time
from typing import List, Dict, Any, Optional

from src.core.config import AppConfig
from src.utils.slack_notifier import send_slack_notification
from src.reporting.shop_detection_reporter import generate_shop_detection_report
from src.reporting.hochbau_detection_reporter import generate_hochbau_detection_report
from src.reporting.exclusion_detection_reporter import generate_exclusion_detection_report
from src.reporting.two_stage_classification_reporter import generate_two_stage_classification_report
from src.reporting.mechanical_engineering_reporter import generate_mechanical_engineering_report

logger = logging.getLogger(__name__)


def generate_all_reports(
    df: pd.DataFrame,
    app_config: AppConfig,
    run_id: str,
    run_output_dir: str,
    run_metrics: Dict[str, Any]
) -> None:
    """
    Orchestrates the generation of all standard pipeline reports.
    """
    logger.info("Starting main report orchestration...")
    report_generation_start_time = time.time()
    report_path = None

    if app_config.pipeline_mode == "shop_detection":
        logger.info("Executing shop_detection reporting pipeline.")
        report_path = generate_shop_detection_report(
            df=df,
            app_config=app_config,
            run_id=run_id,
            run_output_dir=run_output_dir
        )
    elif app_config.pipeline_mode == "hochbau_detection":
        logger.info("Executing hochbau_detection reporting pipeline.")
        report_path = generate_hochbau_detection_report(
            df=df,
            app_config=app_config,
            run_id=run_id,
            run_output_dir=run_output_dir
        )
    elif app_config.pipeline_mode == "exclusion_detection":
        logger.info("Executing exclusion_detection reporting pipeline.")
        report_path = generate_exclusion_detection_report(
            df=df,
            app_config=app_config,
            run_id=run_id,
            run_output_dir=run_output_dir
        )
    elif app_config.pipeline_mode == "two_stage_classification":
        logger.info("Executing two_stage_classification reporting pipeline.")
        report_path = generate_two_stage_classification_report(
            df=df,
            app_config=app_config,
            run_id=run_id,
            run_output_dir=run_output_dir
        )
    elif app_config.pipeline_mode == "mechanical_engineering_detection":
        logger.info("Executing mechanical_engineering_detection reporting pipeline.")
        report_path = generate_mechanical_engineering_report(
            df=df,
            app_config=app_config,
            run_id=run_id,
            run_output_dir=run_output_dir
        )
    else:
        logger.warning(f"No specific reporter for pipeline_mode '{app_config.pipeline_mode}'. No report generated.")

    run_metrics["tasks"]["report_orchestration_duration_seconds"] = round(time.time() - report_generation_start_time, 2)
    
    if app_config.enable_slack_notifications and report_path and app_config.slack_bot_token and app_config.slack_channel_id:
        logger.info("Sending Slack notification...")
        message = (
            f"Shop Confirmation System Pipeline Run Complete\n"
            f"--------------------------------------------\n"
            f"Mode: `{app_config.pipeline_mode}`\n"
            f"Input File: `{os.path.basename(app_config.input_excel_file_path)}`\n"
            f"Rows Processed: `{len(df)}`\n"
            f"Run ID: `{run_id}`\n"
            f"--------------------------------------------\n"
            f"Report is attached."
        )
        send_slack_notification(
            token=app_config.slack_bot_token,
            channel_id=app_config.slack_channel_id,
            message=message,
            file_path=report_path
        )

    logger.info("Main report orchestration finished.")
