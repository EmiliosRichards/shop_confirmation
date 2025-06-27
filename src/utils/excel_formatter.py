import pandas as pd
from typing import List, Optional

def save_df_to_formatted_excel(df: pd.DataFrame, output_path: str, columns_to_drop: Optional[List[str]] = None):
    """
    Saves a DataFrame to an Excel file with auto-adjusted column widths and a table format.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        output_path (str): The path to save the Excel file to.
        columns_to_drop (List[str], optional): A list of columns to drop before saving. Defaults to None.
    """
    df_to_report = df.copy()

    if columns_to_drop:
        for col in columns_to_drop:
            if col in df_to_report.columns:
                df_to_report = df_to_report.drop(columns=col)

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df_to_report.to_excel(writer, sheet_name='Report', index=False)
        
        workbook = writer.book
        worksheet = writer.sheets['Report']

        # Create a table
        (max_row, max_col) = df_to_report.shape
        column_settings = [{'header': column} for column in df_to_report.columns]
        worksheet.add_table(0, 0, max_row, max_col - 1, {'columns': column_settings})

        # Auto-adjust column widths
        for i, col in enumerate(df_to_report.columns):
            column_len = max(df_to_report[col].astype(str).map(len).max(), len(col))
            worksheet.set_column(i, i, column_len + 2)