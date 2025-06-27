import pandas as pd
import os
import logging
from typing import Dict, Any, Optional

from src.core.config import AppConfig
from src.utils.excel_formatter import save_df_to_formatted_excel

logger = logging.getLogger(__name__)

def generate_hochbau_detection_report(
    df: pd.DataFrame,
    app_config: AppConfig,
    run_id: str,
    run_output_dir: str
) -> Optional[str]:
    """
    Generates a simple Excel report for the hochbau detection pipeline.

    This report is a replica of the input file with the additional 
    hochbau detection columns ('is_hochbau', 'hochbau_confidence', 'hochbau_evidence').

    Args:
        df: The final DataFrame containing the processing results.
        app_config: The application configuration object.
        run_id: The unique identifier for the current pipeline run.
        run_output_dir: The directory where the report will be saved.
    """
    try:
        profile = app_config.CLASSIFICATION_PROFILES.get("hochbau_detection", {})
        output_filename = profile.get("output_filename_template", "Hochbau_Detection_Report_{run_id}.xlsx").format(run_id=run_id)
        output_path = os.path.join(run_output_dir, output_filename)

        columns_to_drop = [
            'NormalizedGivenPhoneNumber', 'Primary_Number_1', 'Primary_Type_1', 
            'Primary_SourceURL_1', 'Secondary_Number_1', 'Secondary_Type_1', 
            'Secondary_SourceURL_1', 'Secondary_Number_2', 'Secondary_Type_2', 
            'Secondary_SourceURL_2', 'RunID', 'TargetCountryCodes'
        ]

        save_df_to_formatted_excel(df, output_path, columns_to_drop)
        
        logger.info(f"Hochbau detection report successfully generated at: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to generate hochbau detection report. Error: {e}", exc_info=True)
        return None