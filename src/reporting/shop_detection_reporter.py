import pandas as pd
import os
import logging
from typing import Dict, Any

from src.core.config import AppConfig

logger = logging.getLogger(__name__)

def generate_shop_detection_report(
    df: pd.DataFrame,
    app_config: AppConfig,
    run_id: str,
    run_output_dir: str
) -> None:
    """
    Generates a simple CSV report for the shop detection pipeline.

    This report is a replica of the input file with the additional 
    shop detection columns ('is_shop', 'is_shop_confidence', 'is_shop_evidence').

    Args:
        df: The final DataFrame containing the processing results.
        app_config: The application configuration object.
        run_id: The unique identifier for the current pipeline run.
        run_output_dir: The directory where the report will be saved.
    """
    try:
        output_filename = app_config.shop_detection_output_filename_template.format(run_id=run_id)
        output_path = os.path.join(run_output_dir, output_filename)

        # Ensure all original and new columns are included
        # The final DataFrame should already contain all necessary columns.
        
        # Columns to remove from the final report, as requested
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

        df_to_report.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"Shop detection report successfully generated at: {output_path}")

    except Exception as e:
        logger.error(f"Failed to generate shop detection report. Error: {e}", exc_info=True)