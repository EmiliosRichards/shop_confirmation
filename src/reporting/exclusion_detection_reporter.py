import pandas as pd
import os
import logging
from typing import Dict, Any

from src.core.config import AppConfig

logger = logging.getLogger(__name__)

def generate_exclusion_detection_report(
    df: pd.DataFrame,
    app_config: AppConfig,
    run_id: str,
    run_output_dir: str
) -> None:
    """
    Generates a CSV report for the exclusion detection pipeline.

    This report is a replica of the input file with the additional 
    exclusion detection columns ('is_excluded', 'exclusion_category', 'exclusion_reason').

    Args:
        df: The final DataFrame containing the processing results.
        app_config: The application configuration object.
        run_id: The unique identifier for the current pipeline run.
        run_output_dir: The directory where the report will be saved.
    """
    try:
        exclusion_profile = app_config.CLASSIFICATION_PROFILES.get("exclusion_detection")
        if not exclusion_profile:
            logger.error("exclusion_detection profile not found in AppConfig.CLASSIFICATION_PROFILES. Cannot generate report.")
            return

        output_filename = exclusion_profile.get("output_filename_template", "Exclusion_Detection_Report_{run_id}.csv").format(run_id=run_id)
        output_path = os.path.join(run_output_dir, output_filename)

        # The final DataFrame should already contain all necessary columns.
        
        # Columns to remove from the final report for cleanliness
        columns_to_drop = [
            'NormalizedGivenPhoneNumber', 'ScrapingStatus', 'Overall_VerificationStatus',
            'Original_Number_Status', 'Primary_Number_1', 'Primary_Type_1',
            'Primary_SourceURL_1', 'Secondary_Number_1', 'Secondary_Type_1',
            'Secondary_SourceURL_1', 'Secondary_Number_2', 'Secondary_Type_2',
            'Secondary_SourceURL_2', 'RunID', 'TargetCountryCodes'
        ]
        
        # Create a copy to avoid modifying the original DataFrame in place
        df_to_report = df.copy()

        # Drop columns that exist in the DataFrame
        for col in columns_to_drop:
            if col in df_to_report.columns:
                df_to_report = df_to_report.drop(columns=col)

        df_to_report.to_csv(output_path, index=False, encoding='utf-8-sig', sep=';')
        
        logger.info(f"Exclusion detection report successfully generated at: {output_path}")

    except Exception as e:
        logger.error(f"Failed to generate exclusion detection report. Error: {e}", exc_info=True)