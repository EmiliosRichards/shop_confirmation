"""
Pydantic Schemas for Data Validation and Structuring.

This module defines Pydantic models used throughout the Intelligent Prospect
Analyzer application. These schemas ensure data consistency, provide clear
data structures, and enable validation for inputs and outputs of various
pipeline components, especially those interacting with LLMs or generating reports.
"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class AdditionalInformation(BaseModel):
    """
    Represents a piece of additional information, potentially tied to a phone number,
    extracted during an enriched profiling process. This supports the 'additional_info'
    field for Profile 2.
    """
    info_type: str = Field(description="Type of information (e.g., 'email', 'name', 'role', 'department', 'location').")
    value: Any = Field(description="The actual information content. Can be a string, list, or dict depending on info_type for flexibility.")
    associated_number: Optional[str] = Field(default=None, description="The phone number (ideally E.164) this information is associated with, if any.")
    source_context: Optional[str] = Field(default=None, description="Brief context or source snippet where this info was found, aiding traceability.")
    confidence: Optional[float] = Field(default=None, ge=0, le=1, description="Confidence score for this specific piece of information, if available.")


class HomepageContextOutput(BaseModel):
    """
    Defines the structure for homepage context information extracted by an LLM.
    """
    company_name: Optional[str] = Field(default=None, description="The name of the company as identified from the homepage.")
    summary_description: Optional[str] = Field(default=None, description="A brief summary of the company, typically with a character limit (e.g., 250 characters).")
    industry: Optional[str] = Field(default=None, description="The industry of the company as inferred from the homepage.")

class DomainExtractionBundle(BaseModel):
    """
    Represents a bundle of information extracted for a domain,
    including consolidated contact details and homepage context.
    """
    homepage_context: Optional[HomepageContextOutput] = Field(default=None, description="Contextual information extracted from the company's homepage.")




class WebsiteTextSummary(BaseModel):
    """
    Represents the output of an LLM call for website summarization.
    This schema captures essential information derived from summarizing
    the text content of a scraped website, intended as input for further attribute extraction.
    """
    original_url: str = Field(description="The original input URL that was scraped to generate this summary.")
    summary: str = Field(description="Concise summary of key information from the website, relevant for attribute extraction.")
    extracted_company_name_from_summary: Optional[str] = Field(default=None, description="Company name as identified by the LLM from the website content during summarization.")
    key_topics_mentioned: Optional[List[str]] = Field(default_factory=list, description="A list of key topics, services, or products mentioned in the website content, identified during summarization.")
class DetailedCompanyAttributes(BaseModel):
    """
    Structures the output of an LLM call that extracts detailed
    attributes from a website summary.
    """
    input_summary_url: str = Field(description="URL of the company website that the source summary was derived from.")
    b2b_indicator: Optional[bool] = Field(default=None, description="True if the company primarily serves other businesses (B2B), False if primarily private customers (B2C), null if unclear.")
    phone_outreach_suitability: Optional[bool] = Field(default=None, description="True if the company's product/service seems suitable for telephone-based acquisition, False otherwise, null if unclear.")
    target_group_size_assessment: Optional[str] = Field(default=None, description="Qualitative assessment of potential callable target group size (e.g., 'Appears Small', 'Appears Medium', 'Appears Large / >=500 potential', 'Unknown').")
    industry: Optional[str] = Field(default=None, description="Primary industry of the company.")
    products_services_offered: Optional[List[str]] = Field(default_factory=list, description="List of key products or services offered.")
    usp_key_selling_points: Optional[List[str]] = Field(default_factory=list, description="Unique Selling Propositions or key selling points highlighted.")
    customer_target_segments: Optional[List[str]] = Field(default_factory=list, description="Specific customer segments targeted by the company.")
    business_model: Optional[str] = Field(default=None, description="Description of the company's business model (e.g., 'Service-oriented; Project-based consulting', 'SaaS').")
    company_size_indicators_text: Optional[str] = Field(default=None, description="Textual clues or indicators about company size found in the summary (e.g., 'mentions large enterprise clients', 'startup phase').")
    company_size_category_inferred: Optional[str] = Field(default=None, description="Inferred company size category (e.g., 'Startup', 'SME', 'Large Enterprise', 'Unknown/Not Specified').")
    innovation_level_indicators_text: Optional[str] = Field(default=None, description="Textual clues about the company's innovation level or focus (e.g., 'uses innovative workshops', 'AI-supported').")
    website_clarity_notes: Optional[str] = Field(default=None, description="Notes on how clearly the business model and target group are communicated on the website, based on the summary.")
class GoldenPartnerMatchOutput(BaseModel):
    """
    Structures the output of an LLM call that includes comparison
    results against Golden Partners and a generated sales line.
    """
    analyzed_company_url: str = Field(description="The original URL of the company that was analyzed.")
    analyzed_company_attributes: DetailedCompanyAttributes = Field(description="The full set of detailed attributes extracted for the analyzed company in a previous LLM call.")
    summary: Optional[str] = Field(default=None, description="The summary of the analyzed company.")
    match_score: Optional[Union[int, float, str]] = Field(default=None, description="Score (e.g., 0-10, 0.0-1.0, or 'High/Medium/Low') indicating the strength of the match with the closest Golden Partner.")
    match_rationale_features: Optional[List[str]] = Field(default_factory=list, description="List of key shared features or reasons provided by the LLM for the match.")
    phone_sales_line: Optional[str] = Field(default=None, description="A 1-2 sentence phone sales pitch tailored to the analyzed company, generated by the LLM.")
    matched_partner_name: Optional[str] = Field(default=None, description="Name of the Golden Partner identified by the LLM as the closest match.")
    matched_partner_description: Optional[str] = Field(default=None, description="The summary of the matched golden partner.")

class ShopClassificationOutput(BaseModel):
    """
    Structures the output of an LLM call for classifying if a website is a shop.
    """
    is_shop: Optional[bool] = Field(default=None, description="True if the website is classified as a shop, False otherwise.")
    confidence: Optional[float] = Field(default=None, ge=0, le=1, description="The confidence score of the classification.")
    evidence: Optional[str] = Field(default=None, description="The evidence or rationale for the classification.")