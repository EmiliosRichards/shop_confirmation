"""
Configuration module for the Intelligent Prospect Analyzer & Sales Insights Generator.

This module defines the `AppConfig` class, which centralizes all application
configurations. Settings are loaded from environment variables (typically via a
`.env` file) with sensible defaults provided for most parameters. This approach
allows for easy customization of the application's behavior without modifying
the codebase.

The configuration covers various aspects of the pipeline, including:
- Web scraping parameters (user agents, timeouts, retry logic, link prioritization).
- Output directory structures and filename conventions.
- Large Language Model (LLM) settings (API keys, model names, generation parameters).
- Paths to prompt files for different LLM tasks.
- Input data handling (file paths, column mapping profiles, row processing ranges).
- Logging levels for file and console outputs.
- Keywords for page type classification.
"""
import os
from dotenv import load_dotenv
from typing import List, Optional

# Load environment variables from a .env file.
# Attempts to load from paths relative to this file's location to support
# execution from different project directory levels (e.g., src/core, src, project root).
dotenv_path_1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env')  # Project_Root/.env
dotenv_path_2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '.env') # Workspace_Root/.env (if project is nested)
dotenv_path_project_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env') # Fallback for main_pipeline.py at project root

loaded_env = False
if os.path.exists(dotenv_path_1):
    load_dotenv(dotenv_path_1)
    loaded_env = True
elif os.path.exists(dotenv_path_2):
    load_dotenv(dotenv_path_2)
    loaded_env = True
elif os.path.exists(dotenv_path_project_root):
    load_dotenv(dotenv_path_project_root)
    loaded_env = True
else:
    print(
        f"Warning: .env file not found at {dotenv_path_1}, {dotenv_path_2}, "
        f"or {dotenv_path_project_root}. Using default configurations or "
        "expecting environment variables to be set externally."
    )


class AppConfig:
    """
    Manages application configurations, loading settings primarily from environment
    variables defined in a .env file.

    This class centralizes all configurable parameters for the prospect analysis
    pipeline, including scraper settings, output directories, LLM parameters,
    prompt paths, data handling specifics, and logging levels. It provides
    default values for most settings if they are not specified in the environment.

    Attributes:
        user_agent (str): User-agent string for web scraping.
        default_page_timeout (int): Default timeout for page operations in milliseconds.
        default_navigation_timeout (int): Default timeout for navigation actions in milliseconds.
        scrape_max_retries (int): Maximum retries for a failed scrape attempt.
        scrape_retry_delay_seconds (int): Delay in seconds between scrape retries.
        
        target_link_keywords (List[str]): Keywords to identify relevant internal links.
        scraper_critical_priority_keywords (List[str]): Keywords for top-priority pages.
        scraper_high_priority_keywords (List[str]): Keywords for high-priority pages.
        scraper_max_keyword_path_segments (int): Max path segments for priority keywords.
        scraper_exclude_link_path_patterns (List[str]): URL path patterns to exclude.
        scraper_max_pages_per_domain (int): Max pages to scrape per domain (0 for no limit).
        scraper_min_score_to_queue (int): Minimum score for a link to be queued.
        scraper_score_threshold_for_limit_bypass (int): Score to bypass page limit.
        scraper_max_high_priority_pages_after_limit (int): Max high-priority pages after limit.
        scraper_pages_for_summary_count (int): Number of top pages for summary text.
        
        max_depth_internal_links (int): Maximum depth for following internal links.
        scraper_networkidle_timeout_ms (int): Playwright networkidle timeout (ms).
        
        output_base_dir (str): Base directory for output files.
        scraped_content_subdir (str): Subdirectory for scraped content.
        llm_context_subdir (str): Subdirectory for LLM context/raw responses.
        filename_company_name_max_len (int): Max length for company name in filenames.
        filename_url_domain_max_len (int): Max length for domain in filenames.
        filename_url_hash_max_len (int): Max length for URL hash in filenames.
        
        respect_robots_txt (bool): Whether to respect robots.txt.
        robots_txt_user_agent (str): User-agent for checking robots.txt.
        
        gemini_api_key (Optional[str]): API key for Google Gemini.
        llm_model_name (str): Google Gemini model to use.
        llm_temperature_default (float): Default LLM temperature for response generation.
        llm_temperature_sales_insights (float): LLM temperature for sales insights generation.
        llm_max_tokens (int): Maximum tokens for LLM response.
        llm_chunk_processor_max_tokens (int): Max tokens for LLM chunk processor.
        llm_max_chunks_per_url (int): Maximum number of chunks to process per URL.
        llm_top_k (Optional[int]): LLM top_k sampling parameter.
        llm_top_p (Optional[float]): LLM top_p (nucleus) sampling parameter.
        LLM_MAX_INPUT_CHARS_FOR_SUMMARY (int): Max input characters for summary LLM call.
        llm_max_tokens_summary (Optional[int]): Max tokens for summary generation.
        llm_temperature_summary (Optional[float]): Temperature for summary generation.
        
        PROMPT_PATH_WEBSITE_SUMMARIZER (str): Path to website summarizer prompt.
        prompt_path_summarization (str): Path to the (old) summarization prompt.
        prompt_path_homepage_context (str): Path to homepage context prompt.
        PROMPT_PATH_ATTRIBUTE_EXTRACTOR (str): Path to attribute extractor prompt.
        PROMPT_PATH_COMPARISON_SALES_LINE (str): Path to comparison sales line prompt.
        MAX_GOLDEN_PARTNERS_IN_PROMPT (int): Max golden partners to include in prompts.
        extraction_profile (str): Current extraction profile to use (e.g., "minimal").

        url_probing_tlds (List[str]): TLDs for domain-like input probing.
        enable_dns_error_fallbacks (bool): Enable DNS error fallback strategies.
        
        input_excel_file_path (str): Path to the input data file.
        input_file_profile_name (str): Name of the input column mapping profile.
        INPUT_COLUMN_PROFILES (dict): Available input column mapping profiles.
        output_excel_file_name_template (str): Template for the main summary report Excel file.
        PROSPECT_ANALYSIS_CSV_FILENAME_TEMPLATE (str): Template for prospect analysis CSV.
        skip_rows_config (Optional[int]): Rows to skip from input file start (0-indexed).
        nrows_config (Optional[int]): Rows to read after skipping (None for all).
        consecutive_empty_rows_to_stop (int): Consecutive empty rows to stop processing.
        PATH_TO_GOLDEN_PARTNERS_DATA (str): Path to the Golden Partners data file (CSV or Excel).
        
        log_level (str): Logging level for the file log (e.g., INFO, DEBUG).
        console_log_level (str): Logging level for console output.
 
        page_type_keywords_about (List[str]): Keywords for 'about' pages.
        page_type_keywords_product_service (List[str]): Keywords for 'product/service' pages.
 
    Methods:
        __init__(): Initializes AppConfig by loading values from environment
                    variables or using defaults.
    """

    INPUT_COLUMN_PROFILES = {
        "default": {  # Kept for fallback, new profiles are more specific
            "Unternehmen": "CompanyName",
            "Webseite": "GivenURL",
            "Beschreibung": "Description"
        },
        "prospect_analyzer_input": {
            "Firmennummer": "CompanyID",          # Internal standard name
            "Firma Kurzname": "CompanyNameShort", # Internal standard name
            "Firma Vollname": "CompanyName",      # Maps to existing CompanyName
            "Homepage": "GivenURL",               # CRITICAL: Maps to existing GivenURL
            "Beschreibung": "Description",        # Maps to existing Description
            "Kategorie": "Industry"               # Maps to a general industry field
            # Other columns like 'Strasse', 'PLZ', 'Ort', 'Land' can be added if needed later.
            # No phone number column specified in the new prospect input structure for direct mapping.
        },
        "ManauvKlaus": {
            "firma": "CompanyName",
            "url": "GivenURL"
        },
        "lean_formatted": {
            "Company Name": "CompanyName",
            "URL": "GivenURL"
        },
        "template": {
            "Company Name": "CompanyName",
            "URL": "GivenURL",
            "Contact Person": "ContactPerson",
            "Email": "Email",
            "Industry": "Industry",             # Corrected to match case of prospect_analyzer_input
            "Company Description": "Description"
        },
        "final_80k": {
            "Company": "CompanyName",
            "Website": "GivenURL",
            "Combined_Description": "Description",
            "Industry_Category_Standardized": "Industry"
        },
        "german_standard": {
            "firma": "CompanyName",
            "url": "GivenURL",
            "beschreibung": "Description",
            "kategorie": "Industry"
        },
        "hochbau_profile": {
            "Name": "CompanyName",
            "Address": "Address",
            "Zip": "Zip",
            "City": "City",
            "Phone": "Phone",
            "Fax": "Fax",
            "E-Mail": "Email",
            "URL": "GivenURL"
        },
        "mechanical_engineering_input": {
            "Name": "CompanyName",
            "Address": "Address",
            "Zip": "Zip",
            "City": "City",
            "Phone": "Phone",
            "Fax": "Fax",
            "E-Mail": "Email",
            "URL": "GivenURL"
        }
    }

    def __init__(self):
        # Add new profile for movepay
        self.INPUT_COLUMN_PROFILES["movepay_profile"] = {
            "firma": "CompanyName",
            "url": "GivenURL",
        }
        """
        Initializes the AppConfig instance.

        Loads all configuration parameters from environment variables using `os.getenv()`.
        If an environment variable is not set for a particular configuration, a
        predefined default value is used. It also handles type conversions for
        parameters like integers, floats, booleans, and lists from their string
        representations in environment variables.
        """
        # --- Scraper Configuration ---
        self.user_agent: str = os.getenv('SCRAPER_USER_AGENT', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        self.default_page_timeout: int = int(os.getenv('SCRAPER_PAGE_TIMEOUT_MS', '30000'))
        self.default_navigation_timeout: int = int(os.getenv('SCRAPER_NAVIGATION_TIMEOUT_MS', '60000'))
        self.scrape_max_retries: int = int(os.getenv('SCRAPER_MAX_RETRIES', '2'))
        self.scrape_retry_delay_seconds: int = int(os.getenv('SCRAPER_RETRY_DELAY_SECONDS', '5'))
        
        # Link Prioritization and Control Settings
        target_link_keywords_str: str = os.getenv('TARGET_LINK_KEYWORDS', 'about,company,services,products,solutions,team,mission')
        self.target_link_keywords: List[str] = [kw.strip().lower() for kw in target_link_keywords_str.split(',') if kw.strip()]
        
        critical_priority_keywords_str: str = os.getenv('SCRAPER_CRITICAL_PRIORITY_KEYWORDS', 'about-us,company-profile')
        self.scraper_critical_priority_keywords: List[str] = [kw.strip().lower() for kw in critical_priority_keywords_str.split(',') if kw.strip()]
        
        high_priority_keywords_str: str = os.getenv('SCRAPER_HIGH_PRIORITY_KEYWORDS', 'services,products,solutions')
        self.scraper_high_priority_keywords: List[str] = [kw.strip().lower() for kw in high_priority_keywords_str.split(',') if kw.strip()]
        
        self.scraper_max_keyword_path_segments: int = int(os.getenv('SCRAPER_MAX_KEYWORD_PATH_SEGMENTS', '3'))
        
        exclude_link_patterns_str: str = os.getenv('SCRAPER_EXCLUDE_LINK_PATH_PATTERNS', '/media/,/blog/,/wp-content/,/video/,/hilfe-video/')
        self.scraper_exclude_link_path_patterns: List[str] = [p.strip().lower() for p in exclude_link_patterns_str.split(',') if p.strip()]
        
        self.scraper_max_pages_per_domain: int = int(os.getenv('SCRAPER_MAX_PAGES_PER_DOMAIN', '20'))  # Default 20, 0 for no limit
        self.scraper_min_score_to_queue: int = int(os.getenv('SCRAPER_MIN_SCORE_TO_QUEUE', '40'))
        self.scraper_score_threshold_for_limit_bypass: int = int(os.getenv('SCRAPER_SCORE_THRESHOLD_FOR_LIMIT_BYPASS', '80'))
        self.scraper_max_high_priority_pages_after_limit: int = int(os.getenv('SCRAPER_MAX_HIGH_PRIORITY_PAGES_AFTER_LIMIT', '5'))  # Default to 5

        self.scraper_pages_for_summary_count: int = int(os.getenv('SCRAPER_PAGES_FOR_SUMMARY_COUNT', '3'))
 
        # Existing Scraper Settings
        self.max_depth_internal_links: int = int(os.getenv('MAX_DEPTH_INTERNAL_LINKS', '1'))
        scraper_timeout_str = os.getenv('SCRAPER_NETWORKIDLE_TIMEOUT_MS', '3000').split('#')[0].strip().strip('\'"')
        self.scraper_networkidle_timeout_ms: int = int(scraper_timeout_str)
 
        # --- Output Configuration ---
        self.output_base_dir: str = os.getenv('OUTPUT_BASE_DIR', 'output_data')  # Relative to project root
        self.scraped_content_subdir: str = 'scraped_content'
        self.llm_context_subdir: str = 'llm_context'  # Subdirectory for LLM raw responses
        self.filename_company_name_max_len: int = int(os.getenv('FILENAME_COMPANY_NAME_MAX_LEN', '25'))  # Default to 25
        self.filename_url_domain_max_len: int = int(os.getenv('FILENAME_URL_DOMAIN_MAX_LEN', '8'))    # Default to 8
        self.filename_url_hash_max_len: int = int(os.getenv('FILENAME_URL_HASH_MAX_LEN', '8'))        # Default to 8

        # --- Robots.txt Handling ---
        self.respect_robots_txt: bool = os.getenv('RESPECT_ROBOTS_TXT', 'True').lower() == 'true'
        self.robots_txt_user_agent: str = os.getenv('ROBOTS_TXT_USER_AGENT', '*')

        # --- LLM Configuration ---
        self.gemini_api_key: Optional[str] = os.getenv('GEMINI_API_KEY')
        self.llm_model_name: str = os.getenv('LLM_MODEL_NAME', 'gemini-1.5-pro-latest')  # Default to a capable model
        self.llm_temperature_default: float = float(os.getenv('LLM_TEMPERATURE_DEFAULT', '0.3'))
        self.llm_max_tokens: int = int(os.getenv('LLM_MAX_TOKENS', '3000'))  # Updated default
        self.llm_chunk_processor_max_tokens: int = int(os.getenv('LLM_CHUNK_PROCESSOR_MAX_TOKENS', '4096'))
        self.llm_max_chunks_per_url: int = int(os.getenv('LLM_MAX_CHUNKS_PER_URL', '5'))

        llm_top_k_str = os.getenv('LLM_TOP_K')
        self.llm_top_k: Optional[int] = int(llm_top_k_str) if llm_top_k_str and llm_top_k_str.isdigit() and int(llm_top_k_str) > 0 else None
        
        llm_top_p_str = os.getenv('LLM_TOP_P')
        self.llm_top_p: Optional[float] = float(llm_top_p_str) if llm_top_p_str else None
        try:
            if self.llm_top_p is not None and not (0.0 <= self.llm_top_p <= 1.0):
                print(f"Warning: LLM_TOP_P value '{self.llm_top_p}' is outside the valid range [0.0, 1.0]. It will be ignored.")
                self.llm_top_p = None
        except ValueError:
            print(f"Warning: Invalid LLM_TOP_P value '{llm_top_p_str}'. It will be ignored.")
            self.llm_top_p = None

        # Specific LLM settings for summary, overriding general ones if provided
        llm_max_tokens_summary_str = os.getenv('LLM_MAX_TOKENS_SUMMARY')
        self.llm_max_tokens_summary: Optional[int] = int(llm_max_tokens_summary_str) if llm_max_tokens_summary_str and llm_max_tokens_summary_str.isdigit() else None
        
        llm_temperature_summary_str = os.getenv('LLM_TEMPERATURE_SUMMARY')
        self.llm_temperature_summary: Optional[float] = None
        if llm_temperature_summary_str:
            try:
                temp_val = float(llm_temperature_summary_str)
                if 0.0 <= temp_val <= 2.0:  # Common range for temperature
                    self.llm_temperature_summary = temp_val
                else:
                    print(f"Warning: LLM_TEMPERATURE_SUMMARY value '{temp_val}' is outside the typical range [0.0, 2.0]. It will be ignored.")
            except ValueError:
                print(f"Warning: Invalid LLM_TEMPERATURE_SUMMARY value '{llm_temperature_summary_str}'. It will be ignored.")

        # --- Extraction Profiles and Prompt Paths ---
        # --- Extraction Profiles and Prompt Paths ---
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        def get_clean_path(env_var: str, default_path: str) -> str:
            raw_path = os.getenv(env_var, default_path)
            # Normalize path separators for consistent checking
            raw_path = raw_path.replace('\\', '/')
            project_root_name = os.path.basename(project_root)
            
            # Check if the raw_path starts with the project root directory name and a slash
            if raw_path.startswith(f"{project_root_name}/"):
                # Strip the redundant prefix
                raw_path = raw_path[len(project_root_name) + 1:]
            
            return os.path.join(project_root, raw_path)

        self.prompt_path_summarization: str = get_clean_path('PROMPT_PATH_SUMMARIZATION', 'prompts/summarization_prompt.txt')
        self.extraction_profile: str = os.getenv('EXTRACTION_PROFILE', "minimal")
        self.prompt_path_homepage_context: str = get_clean_path('PROMPT_PATH_HOMEPAGE_CONTEXT', 'prompts/homepage_context_prompt.txt')
        self.LLM_MAX_INPUT_CHARS_FOR_SUMMARY: int = int(os.getenv('LLM_MAX_INPUT_CHARS_FOR_SUMMARY', '40000'))
 
        # --- URL Probing Configuration ---
        url_probing_tlds_str: str = os.getenv('URL_PROBING_TLDS', 'de,com,at,ch')
        self.url_probing_tlds: List[str] = [tld.strip().lower() for tld in url_probing_tlds_str.split(',') if tld.strip()]
        self.enable_dns_error_fallbacks: bool = os.getenv('ENABLE_DNS_ERROR_FALLBACKS', 'True').lower() == 'true'

        # --- Data Handling & Input Profiling ---
        self.input_excel_file_path: str = os.getenv('INPUT_EXCEL_FILE_PATH', 'data_to_be_inputed.xlsx')  # Relative to project root
        self.input_file_profile_name: str = os.getenv("INPUT_FILE_PROFILE_NAME", "movepay_profile")
        self.csv_delimiter: str = os.getenv("CSV_DELIMITER", ";")
        self.output_excel_file_name_template: str = os.getenv('OUTPUT_EXCEL_FILE_NAME_TEMPLATE', 'Pipeline_Summary_Report_{run_id}.xlsx')
        self.PATH_TO_GOLDEN_PARTNERS_DATA: str = os.getenv('PATH_TO_GOLDEN_PARTNERS_DATA', 'data/kunden_golden_standard.xlsx')

        # --- Row Processing Range Configuration ---
        self.skip_rows_config: Optional[int] = None
        self.nrows_config: Optional[int] = None
        raw_row_range: Optional[str] = os.getenv('ROW_PROCESSING_RANGE', "")

        if raw_row_range:
            raw_row_range = raw_row_range.strip()
            if not raw_row_range or raw_row_range == "0":
                pass  # Process all rows, skip_rows_config and nrows_config remain None
            elif '-' in raw_row_range:
                parts = raw_row_range.split('-', 1)
                start_str, end_str = parts[0].strip(), parts[1].strip()

                start_val: Optional[int] = None
                end_val: Optional[int] = None

                if start_str and start_str.isdigit():
                    start_val = int(start_str)
                
                if end_str and end_str.isdigit():
                    end_val = int(end_str)

                if start_val is not None and start_val > 0:
                    self.skip_rows_config = start_val - 1  # 0-indexed skip
                    if end_val is not None and end_val >= start_val:
                        self.nrows_config = end_val - start_val + 1
                    elif end_str == "":  # Format "N-" (from N to end)
                        self.nrows_config = None  # Read all after skipping
                    elif end_val is not None and end_val < start_val:
                        print(f"Warning: Invalid ROW_PROCESSING_RANGE '{raw_row_range}'. End value < Start value. Processing all rows.")
                        self.skip_rows_config = None
                        self.nrows_config = None
                elif start_str == "" and end_val is not None and end_val > 0:  # Format "-M" (first M rows)
                    self.skip_rows_config = None  # Or 0, effectively the same for pandas
                    self.nrows_config = end_val
                else:
                    print(f"Warning: Invalid ROW_PROCESSING_RANGE format '{raw_row_range}'. Expected N-M, N-, -M, or N. Processing all rows.")
            elif raw_row_range.isdigit() and int(raw_row_range) > 0:  # Single number "N"
                self.skip_rows_config = None  # Or 0
                self.nrows_config = int(raw_row_range)
            else:
                if raw_row_range != "0":  # "0" is a valid way to say "all rows"
                    print(f"Warning: Invalid ROW_PROCESSING_RANGE value '{raw_row_range}'. Processing all rows.")
        
        # --- Data Handling Enhancements ---
        self.consecutive_empty_rows_to_stop: int = int(os.getenv('CONSECUTIVE_EMPTY_ROWS_TO_STOP', '3'))

        # --- Logging Configuration ---
        self.log_level: str = os.getenv('LOG_LEVEL', 'INFO').upper()
        self.console_log_level: str = os.getenv('CONSOLE_LOG_LEVEL', 'WARNING').upper()

        # --- Pipeline Execution Configuration ---
        self.pipeline_mode: str = os.getenv('PIPELINE_MODE', 'two_stage_classification')

        # --- Classification Profiles ---
        self.CLASSIFICATION_PROFILES = {
            "shop_detection": {
                "prompt_path": get_clean_path('PROMPT_PATH_SHOP_DETECTION', 'prompts/shop_detection_prompt.txt'),
                "output_columns": {
                    "is_shop": "is_shop",
                    "confidence": "is_shop_confidence",
                    "evidence": "is_shop_evidence"
                },
                "target_keywords": [kw.strip().lower() for kw in os.getenv('SHOP_DETECTION_TARGET_KEYWORDS', 'products,shop,store,buy,cart,checkout,pricing,plans').split(',') if kw.strip()],
                "output_filename_template": os.getenv('SHOP_DETECTION_OUTPUT_FILENAME_TEMPLATE', 'Shop_Detection_Report_{run_id}.xlsx')
            },
            "hochbau_detection": {
                "prompt_path": get_clean_path('PROMPT_PATH_HOCHBAU_DETECTION', 'prompts/hochbau_detection_prompt.txt'),
                "output_columns": {
                    "is_hochbau": "is_hochbau",
                    "confidence_score": "hochbau_confidence",
                    "evidence": "hochbau_evidence"
                },
                "target_keywords": [kw.strip().lower() for kw in os.getenv('HOCHBAU_DETECTION_TARGET_KEYWORDS', 'hochbau,tiefbau,bauunternehmen,baugesellschaft,bau,architektur,wohnbau,gewerbebau,industriebau').split(',') if kw.strip()],
                "output_filename_template": os.getenv('HOCHBAU_DETECTION_OUTPUT_FILENAME_TEMPLATE', 'Hochbau_Detection_Report_{run_id}.xlsx')
            },
            "exclusion_detection": {
                "prompt_path": get_clean_path('PROMPT_PATH_EXCLUSION_DETECTION', 'prompts/exclusion_detection_prompt.txt'),
                "output_columns": {
                    "is_excluded": "is_excluded",
                    "exclusion_category": "exclusion_category",
                    "reason": "exclusion_reason"
                },
                "target_keywords": [kw.strip().lower() for kw in os.getenv('EXCLUSION_DETECTION_TARGET_KEYWORDS', 'about,company,services,products,solutions,team,mission,imprint,impressum,kontakt,contact').split(',') if kw.strip()],
                "output_filename_template": os.getenv('EXCLUSION_DETECTION_OUTPUT_FILENAME_TEMPLATE', 'Exclusion_Detection_Report_{run_id}.xlsx')
            },
            "positive_criteria_detection": {
                "prompt_path": get_clean_path('PROMPT_PATH_POSITIVE_CRITERIA_DETECTION', 'prompts/positive_criteria_prompt.txt'),
                "output_columns": {
                    "meets_criteria": "meets_criteria",
                    "matching_category": "matching_category",
                    "matching_evidence": "matching_evidence",
                    "primary_business_focus": "primary_business_focus"
                },
                "target_keywords": [kw.strip().lower() for kw in os.getenv('POSITIVE_CRITERIA_DETECTION_TARGET_KEYWORDS', 'about,company,services,products,solutions,team,mission,imprint,impressum,kontakt,contact').split(',') if kw.strip()],
                "output_filename_template": os.getenv('TWO_STAGE_CLASSIFICATION_OUTPUT_FILENAME_TEMPLATE', 'Two_Stage_Classification_Report_{run_id}.xlsx')
            },
            "mechanical_engineering_detection": {
                "prompt_path": get_clean_path('PROMPT_PATH_MECH_ENG_DETECTION', 'prompts/mechanical_engineering_prompt.txt'),
                "output_columns": {
                    "is_mech": "is_mech",
                    "mech_reasoning": "mech_reasoning",
                    "industry": "industry"
                },
                "target_keywords": [kw.strip().lower() for kw in os.getenv('MECH_ENG_DETECTION_TARGET_KEYWORDS', 'maschinenbau,anlagenbau,sonderanlagenbau,automation,automatisierung,robotik,steuerungstechnik,antriebstechnik,fördertechnik,produktionstechnik,fertigungstechnik,werkzeugbau,zerspanung,schweissen,montage,industrie,industrieanlagen').split(',') if kw.strip()],
                "output_filename_template": os.getenv('MECH_ENG_DETECTION_OUTPUT_FILENAME_TEMPLATE', 'Mechanical_Engineering_Report_{run_id}.xlsx')
            }
        }

        # --- Page Type Classification Keywords ---
        page_type_about_str: str = os.getenv('PAGE_TYPE_KEYWORDS_ABOUT', 'about,about-us,company,profile,mission,vision,team')
        self.page_type_keywords_about: List[str] = [kw.strip().lower() for kw in page_type_about_str.split(',') if kw.strip()]

        page_type_product_service_str: str = os.getenv('PAGE_TYPE_KEYWORDS_PRODUCT_SERVICE', 'products,services,solutions,offerings,platform,features')
        self.page_type_keywords_product_service: List[str] = [kw.strip().lower() for kw in page_type_product_service_str.split(',') if kw.strip()]

        # --- Slack Notifications ---
        self.enable_slack_notifications: bool = os.getenv('ENABLE_SLACK_NOTIFICATIONS', 'False').lower() == 'true'
        self.slack_bot_token: Optional[str] = os.getenv('SLACK_BOT_TOKEN')
        self.slack_channel_id: Optional[str] = os.getenv('SLACK_CHANNEL_ID')