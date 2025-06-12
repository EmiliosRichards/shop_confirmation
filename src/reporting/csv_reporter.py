"""
Handles the generation of CSV reports for prospect analysis.

This module provides functions to take structured prospect analysis data,
flatten it, and write it to a CSV file in a specified output directory.
It includes logic for handling nested data structures and ensuring
consistent output for easier consumption and review.
"""
import os
import logging
from typing import List, Dict, Any, Optional
import pandas as pd

from ..core.schemas import GoldenPartnerMatchOutput, DetailedCompanyAttributes

logger = logging.getLogger(__name__)


def write_prospect_analysis_to_csv(
    output_data: List[GoldenPartnerMatchOutput],
    output_dir: str,
    filename_template: str,
    run_id: str,
    original_df: pd.DataFrame
) -> Optional[str]:
    """
    Writes the prospect analysis data to a CSV file.

    Args:
        output_data (List[GoldenPartnerMatchOutput]): A list of GoldenPartnerMatchOutput objects.
        output_dir (str): The directory where the CSV file will be saved.
        filename_template (str): The template for the CSV filename
                                 (e.g., "prospect_analysis_report_{run_id}.csv").
        run_id (str): The unique identifier for the current run.
        original_df (pd.DataFrame): The original input DataFrame.

    Returns:
        Optional[str]: The full path to the saved CSV file, or None if an error occurred.
    """
    if not output_data:
        logger.warning("No output data provided to write_prospect_analysis_to_csv. Skipping CSV generation.")
        return None

    try:
        os.makedirs(output_dir, exist_ok=True)
        filename = filename_template.format(run_id=run_id)
        full_path = os.path.join(output_dir, filename)

        report_data = []
        for index, original_row in original_df.iterrows():
            row_output = next((item for item in output_data if item.analyzed_company_url == original_row.get('url')), None)

            if row_output:
                attrs = row_output.analyzed_company_attributes
                row = {
                    'Company Name': original_row.get('firma'),
                    'Number': original_row.get('telefonnummer'),
                    'URL': row_output.analyzed_company_url,
                    'Description': row_output.summary if row_output.summary else original_row.get('beschreibung'),
                    'Industry': attrs.industry if attrs else original_row.get('kategorie'),
                    'Sales Line': row_output.phone_sales_line,
                    'Key Resonating Themes': "; ".join(row_output.match_rationale_features) if row_output.match_rationale_features else "",
                    'Matched Partner Name': '',
                    'Matched Partner Description': row_output.matched_partner_description,
                    'Match Score': row_output.match_score,
                    'B2B Indicator': attrs.b2b_indicator if attrs else '',
                    'Phone Outreach Suitability': attrs.phone_outreach_suitability if attrs else '',
                    'Target Group Size Assessment': attrs.target_group_size_assessment if attrs else '',
                    'Products/Services Offered': "; ".join(attrs.products_services_offered) if attrs and attrs.products_services_offered else '',
                    'USP/Key Selling Points': "; ".join(attrs.usp_key_selling_points) if attrs and attrs.usp_key_selling_points else '',
                    'Customer Target Segments': "; ".join(attrs.customer_target_segments) if attrs and attrs.customer_target_segments else '',
                    'Business Model': attrs.business_model if attrs else '',
                    'Company Size Inferred': attrs.company_size_category_inferred if attrs else '',
                    'Innovation Level Indicators': attrs.innovation_level_indicators_text if attrs else '',
                    'Website Clarity Notes': attrs.website_clarity_notes if attrs else ''
                }
            else:
                row = {
                    'Company Name': original_row.get('firma'),
                    'Number': original_row.get('telefonnummer'),
                    'URL': original_row.get('url'),
                    'Description': original_row.get('beschreibung'),
                    'Industry': original_row.get('kategorie'),
                    'Sales Line': '',
                    'Key Resonating Themes': '',
                    'Matched Partner Name': '',
                    'Matched Partner Description': '',
                    'Match Score': '',
                    'B2B Indicator': '',
                    'Phone Outreach Suitability': '',
                    'Target Group Size Assessment': '',
                    'Products/Services Offered': '',
                    'USP/Key Selling Points': '',
                    'Customer Target Segments': '',
                    'Business Model': '',
                    'Company Size Inferred': '',
                    'Innovation Level Indicators': '',
                    'Website Clarity Notes': ''
                }
            report_data.append(row)

        if not report_data:
            logger.warning(f"No data to write for prospect analysis report. Run ID: {run_id}")
            return None

        df = pd.DataFrame(report_data)
        df.to_csv(full_path, index=False, encoding='utf-8-sig')
        logger.info(f"Successfully wrote prospect analysis to CSV: {full_path}")
        return full_path

    except Exception as e:
        logger.error(f"Error writing prospect analysis to CSV for run_id {run_id}: {e}", exc_info=True)
        return None

def write_sales_outreach_report(
    output_data: List[GoldenPartnerMatchOutput],
    output_dir: str,
    run_id: str,
    original_df: pd.DataFrame,
    golden_partners_raw: List[Dict[str, Any]]
) -> Optional[str]:
    """
    Writes the sales outreach data to a CSV file.

    Args:
        output_data (List[GoldenPartnerMatchOutput]): A list of GoldenPartnerMatchOutput objects.
        output_dir (str): The directory where the CSV file will be saved.
        run_id (str): The unique identifier for the current run.
        original_df (pd.DataFrame): The original input DataFrame.
        golden_partners_raw (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                                  represents a golden partner.

    Returns:
        Optional[str]: The full path to the saved CSV file, or None if an error occurred.
    """
    if not output_data:
        logger.warning("No output data provided to write_sales_outreach_report. Skipping CSV generation.")
        return None

    try:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"SalesOutreachReport_{run_id}.csv"
        full_path = os.path.join(output_dir, filename)

        report_data = []
        

        for index, original_row in original_df.iterrows():
            original_url = original_row.get('GivenURL') # Corrected from 'url' to 'GivenURL'
            row_output = next((item for item in output_data if item.analyzed_company_url == original_url), None)

            if row_output:
                attrs = row_output.analyzed_company_attributes
                row = {
                    'Company Name': original_row.get('CompanyName'),
                    'Number': original_row.get('Telefonnummer'),
                    'URL': row_output.analyzed_company_url,
                    'Description': row_output.summary if row_output.summary else original_row.get('Beschreibung'),
                    'Industry': attrs.industry if attrs else original_row.get('Kategorie'),
                    'Sales Line': row_output.phone_sales_line,
                    'Key Resonating Themes': "; ".join(row_output.match_rationale_features) if row_output.match_rationale_features else "",
                    'Matched Partner Name': row_output.matched_partner_name,
                    'Matched Partner Description': next((p.get('description', '') for p in golden_partners_raw if p.get('name') == row_output.matched_partner_name), '') if row_output.matched_partner_name else '',
                    'Match Score': row_output.match_score,
                    'B2B Indicator': attrs.b2b_indicator if attrs else '',
                    'Phone Outreach Suitability': attrs.phone_outreach_suitability if attrs else '',
                    'Target Group Size Assessment': attrs.target_group_size_assessment if attrs else '',
                    'Products/Services Offered': "; ".join(attrs.products_services_offered) if attrs and attrs.products_services_offered else '',
                    'USP/Key Selling Points': "; ".join(attrs.usp_key_selling_points) if attrs and attrs.usp_key_selling_points else '',
                    'Customer Target Segments': "; ".join(attrs.customer_target_segments) if attrs and attrs.customer_target_segments else '',
                    'Business Model': attrs.business_model if attrs else '',
                    'Company Size Inferred': attrs.company_size_category_inferred if attrs else '',
                    'Innovation Level Indicators': attrs.innovation_level_indicators_text if attrs else '',
                    'Website Clarity Notes': attrs.website_clarity_notes if attrs else ''
                }
            else:
                row = {
                    'Company Name': original_row.get('CompanyName'),
                    'Number': original_row.get('Telefonnummer'),
                    'URL': original_row.get('GivenURL'),
                    'Description': original_row.get('Beschreibung'),
                    'Industry': original_row.get('Kategorie'),
                    'Sales Line': '',
                    'Key Resonating Themes': '',
                    'Matched Partner Name': '',
                    'Matched Partner Description': '',
                    'Match Score': '',
                    'B2B Indicator': '',
                    'Phone Outreach Suitability': '',
                    'Target Group Size Assessment': '',
                    'Products/Services Offered': '',
                    'USP/Key Selling Points': '',
                    'Customer Target Segments': '',
                    'Business Model': '',
                    'Company Size Inferred': '',
                    'Innovation Level Indicators': '',
                    'Website Clarity Notes': ''
                }
            report_data.append(row)

        if not report_data:
            logger.warning(f"No data to write for sales outreach report. Run ID: {run_id}")
            return None

        df = pd.DataFrame(report_data)
        df.to_csv(full_path, index=False, encoding='utf-8-sig')
        logger.info(f"Successfully wrote sales outreach report to CSV: {full_path}")
        return full_path

    except Exception as e:
        logger.error(f"Error writing sales outreach report to CSV for run_id {run_id}: {e}", exc_info=True)
        return None