import pandas as pd
import os
import logging
from typing import Dict, Any

from src.core.config import AppConfig

logger = logging.getLogger(__name__)

def generate_two_stage_classification_report(
    df: pd.DataFrame,
    app_config: AppConfig,
    run_id: str,
    run_output_dir: str
) -> None:
    """
    Generates a comprehensive CSV report for the two-stage classification pipeline.

    This report includes the results from both the exclusion and positive criteria checks.

    Args:
        df: The final DataFrame containing the processing results.
        app_config: The application configuration object.
        run_id: The unique identifier for the current pipeline run.
        run_output_dir: The directory where the report will be saved.
    """
    try:
        output_filename = f"Two_Stage_Classification_Report_{run_id}.csv"
        output_path = os.path.join(run_output_dir, output_filename)

        # The final DataFrame should already contain all necessary columns from both stages.
        
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
        
        logger.info(f"Two-stage classification report successfully generated at: {output_path}")

    except Exception as e:
        logger.error(f"Failed to generate two-stage classification report. Error: {e}", exc_info=True)