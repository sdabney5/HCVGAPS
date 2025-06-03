# -*- coding: utf-8 -*-
"""
HCV Processing Module for Housing Choice Voucher (HCV) Eligibility Analysis.

This module contains the core function that processes HCV eligibility data for a single state-year combination.
The workflow includes:
  1. Loading data from various CSV sources.
  2. Processing counties and allocation using crosswalk data.
  3. Cleaning and splitting the data.
  4. Performing feature engineering.
  5. Calculating HCV eligibility.
  6. Adjusting for incarcerated individuals.
  7. Generating final outputs.

The function expects a configuration dictionary with file paths and processing options,
including keys "state" and "year" that specify the current state and year.
"""

import logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

from .hcv_data_loading import (
    load_ipums_data,
    load_crosswalk_data,
    load_income_limits,
    load_incarceration_df,
    load_hud_hcv_data
)
from .hcv_fill_counties_and_allocation import fill_missing_county_values
from .hcv_income_cleaning_and_household_splitting import (
    clean_single_family_income_data,
    split_multifamily_households,
    process_multi_family_income_data
)
from .hcv_family_feature_engineering import family_feature_engineering, flatten_households_to_single_rows
from .hcv_eligibility_calculation import calculate_hcv_eligibility
from .hcv_prisoner_adjustment import stratified_selection_for_incarcerated_individuals
from .hcv_final_outputs import calculate_voucher_gap_and_save

def process_hcv_eligibility(config):
    """
    Process Housing Choice Voucher (HCV) eligibility data for a single state-year combination.

    Workflow:
      1. Load data from various sources.
      2. Process counties and allocation using crosswalk data.
      3. Clean and split data.
      4. Perform feature engineering.
      5. Calculate HCV eligibility.
      6. Adjust for prisoners.
      7. Generate final outputs.

    Parameters:
        config (dict): A configuration dictionary containing file paths and processing options,
                       including "state" and "year" keys.
    """
    # Load Data: IPUMS data is read from the file path provided in config.
    ipums_df = load_ipums_data(config['ipums_data_path'])
    if ipums_df is None:
        logging.error("Error: Failed to load IPUMS data. Exiting script.")
        exit(1)
    logging.info("Loaded IPUMS data")

    # Load Crosswalk Data
    crosswalk_2012_df, crosswalk_2022_df = load_crosswalk_data(
        config['crosswalk_2012_path'], config['crosswalk_2022_path']
    )
    if crosswalk_2012_df is None or crosswalk_2022_df is None:
        logging.error("Error: Failed to load crosswalk data. Exiting script.")
        exit(1)
    logging.info("Loaded 2012 and 2022 crosswalk data")

    # Load Income Limits Data
    income_limits_df = load_income_limits(config['income_limits_path'])
    if income_limits_df is None:
        logging.error("Error: Failed to load income limits data. Exiting script.")
        exit(1)
    logging.info("Loaded income limits data")
    
    # Load Incarceration Data
    incarceration_df = load_incarceration_df(config['incarceration_data_path'])
    if incarceration_df is None:
        logging.error("Error: Failed to load incarceration data. Exiting script.")
        exit(1)
    logging.info("Loaded incarceration data")

    # ── DEBUG: inspect the DataFrame before prisoner adjustment ────────
    logging.info("DEBUG: incarceration_df.shape = %s", incarceration_df.shape)
    logging.info("DEBUG: incarceration_df.dtypes:\n%s", incarceration_df.dtypes)
    logging.info("DEBUG: incarceration_df.head():\n%s", incarceration_df.head().to_string())
    if 'County_Name' in incarceration_df.columns:
        logging.info("DEBUG: unique County_Name values: %s", incarceration_df['County_Name'].unique()[:10])
    else:
        logging.warning("DEBUG: 'County_Name' column not found in incarceration_df")

    # Load HUD HCV Data
    hud_hcv_df = load_hud_hcv_data(config['hud_hcv_data_path'])
    if hud_hcv_df is None:
        logging.error("Error: Failed to load HUD HCV data. Exiting script.")
        exit(1)
    logging.info("Loaded HUD HCV data")
    
    # Process Counties and Allocation
    ipums_df = fill_missing_county_values(ipums_df, crosswalk_2012_df, crosswalk_2022_df)
    logging.info("Complete: processed missing county values")
    
    # Clean and Split Data
    ipums_df = clean_single_family_income_data(ipums_df)
    logging.info("Complete: clean single-family income data")

    ipums_df = split_multifamily_households(ipums_df)
    logging.info("Complete: split multifamily households")

    ipums_df = process_multi_family_income_data(ipums_df)
    logging.info("Complete: process multi-family income data")

    # Feature Engineering
    ipums_df = family_feature_engineering(ipums_df)
    logging.info("Complete: perform family feature engineering")

    ipums_df = flatten_households_to_single_rows(ipums_df)
    logging.info("Complete: flatten households to single rows")

    # Eligibility Calculation
    ipums_df = calculate_hcv_eligibility(ipums_df, income_limits_df)
    logging.info("Complete: calculate HCV eligibility")

    # Adjust for Prisoners
    ipums_df = stratified_selection_for_incarcerated_individuals(
        ipums_df,
        incarceration_df,
        config['prisoners_identified_by_GQTYPE2'],
        config['race_sampling'],
        config['verbose']
    )
    logging.info("Complete: adjusted for prisoners")

    # Final Outputs:
    calculate_voucher_gap_and_save(
        ipums_df,
        hud_hcv_df,
        config['output_directory'],
        config['state'],
        config['year'],
        display_race_stats=config['display_race_stats']
    )
