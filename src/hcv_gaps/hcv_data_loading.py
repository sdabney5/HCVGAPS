"""
This module contains functions for loading required datasets.
Each function performs checks.

Functions:

1. load_ipums_data(filepath):
    - Loads IPUMS data from a CSV and checks for required columns and missing values.

2. load_crosswalk_data(filepath):
    - Loads MCDC crosswalk data from a CSV and performs checks.

3. load_income_limits(filepath):
    - Loads income limits data from a CSV and checks for the HUD API naming convention.

4. load_incarceration_df(filepath):
    - Loads and validates the incarceration dataset from a CSV; checks for required columns and missing values.

5. load_hud_hcv_data(filepath):
    - Loads HUD Picture of Subsidized Housing data from a CSV and checks for required columns and missing values.

Usage:
To use, import this module and call the loading functions with the correct file paths.

Example:
    import hcv_data_loading as hcv_data

    ipums_df = hcv_data.load_ipums_data('path_to_ipums_data.csv')
    crosswalk_df = hcv_data.load_crosswalk_data('path_to_crosswalk_data.csv')
    income_limits_df = hcv_data.load_income_limits('path_to_income_limits.csv')
    incarceration_df = hcv_data.load_incarceration_df('path_to_incarceration_data.csv')
    hud_hcv_df = hcv_data.load_hud_hcv_data('path_to_hud_hcv_data.csv')
"""

#imports
import pandas as pd
import logging

logging.info("This is a log message from hcv_data_loading.py")


def load_ipums_data(filepath):
    """
    Load IPUMS data from a CSV file and perform variable checks.

    Parameters:
    filepath (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Loaded IPUMS data (if all checks pass).

    Raises:
    ValueError: If issues are found.
    """
    try:
        # Load the dataset
        ipums_df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Error loading IPUMS file: {e}")

    # Check for required columns
    required_columns = ['PUMA', 'COUNTYICP', 'HHINCOME', 'FTOTINC', 'INCWAGE', 'INCSS', 'INCWELFR', 'HHWT',
                        'INCINVST', 'INCRETIR', 'INCSUPP', 'INCEARN', 'INCOTHER', 'NFAMS', 'FAMUNIT', 'CBSERIAL']
    missing_columns = [col for col in required_columns if col not in ipums_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}. "
                         "Make sure the IPUMS dataset includes all necessary variables.")

    # Check for missing values
    missing_pumas = ipums_df['PUMA'].isnull().sum()
    missing_countyicp = ipums_df['COUNTYICP'].isnull().sum()
    missing_income_columns = {col: ipums_df[col].isnull().sum() for col in required_columns if ipums_df[col].isnull().sum() > 0}

    if missing_pumas > 0 or missing_countyicp > 0 or missing_income_columns:
        error_message = [f"Number of rows with missing PUMA values: {missing_pumas}",
                         f"Number of rows with missing COUNTYICP values: {missing_countyicp}"]
        for col, count in missing_income_columns.items():
            error_message.append(f"Number of rows with missing {col} values: {count}")
        raise ValueError(" | ".join(error_message) + ". Ensure there are no missing values in columns.")

    # Convert columns to appropriate types and clean string fields
    try:
        # Convert PUMA and COUNTYICP to strings and strip whitespace
        ipums_df['PUMA'] = ipums_df['PUMA'].astype(str).str.strip()
        ipums_df['COUNTYICP'] = ipums_df['COUNTYICP'].astype(str).str.strip()
        ipums_df['HHWT'] = ipums_df['HHWT'].astype(float)
        income_columns = ['HHINCOME', 'FTOTINC', 'INCWAGE', 'INCSS', 'INCWELFR', 'INCINVST', 'INCRETIR', 'INCSUPP', 'INCEARN', 'INCOTHER']
        ipums_df[income_columns] = ipums_df[income_columns].apply(pd.to_numeric, errors='coerce')
    except ValueError as e:
        raise ValueError(f"Error converting columns to appropriate types: {e}")

    logging.info("IPUMS data loaded successfully and all checks passed.")
    return ipums_df


def load_crosswalk_data(filepath_2012, filepath_2022):
    """
    Load, clean, and normalize the 2012 and 2022 MCDC crosswalk datasets.

    Parameters:
    filepath_2012 (str): Path to the 2012 crosswalk CSV file.
    filepath_2022 (str): Path to the 2022 crosswalk CSV file.

    Returns:
    tuple: Two cleaned and normalized DataFrames (crosswalk_2012_df, crosswalk_2022_df).

    Raises:
    ValueError: If the files have missing required columns, missing values, or invalid data types.
    """
    try:
        # Load the datasets
        crosswalk_2012_df = pd.read_csv(filepath_2012)
        crosswalk_2022_df = pd.read_csv(filepath_2022)

        # Define required columns
        required_columns = ['State code', 'PUMA', 'County code', 'State abbr.', 'County_Name', 'allocation factor']

        # Validate 2012 dataset
        missing_columns_2012 = [col for col in required_columns if col not in crosswalk_2012_df.columns]
        if missing_columns_2012:
            raise ValueError(f"2012 crosswalk dataset is missing required columns: {missing_columns_2012}")
        if crosswalk_2012_df.isnull().any().any():
            missing_details_2012 = crosswalk_2012_df.isnull().sum()
            raise ValueError(f"2012 crosswalk dataset contains missing values:\n{missing_details_2012}")

        # Validate 2022 dataset
        missing_columns_2022 = [col for col in required_columns if col not in crosswalk_2022_df.columns]
        if missing_columns_2022:
            raise ValueError(f"2022 crosswalk dataset is missing required columns: {missing_columns_2022}")
        if crosswalk_2022_df.isnull().any().any():
            missing_details_2022 = crosswalk_2022_df.isnull().sum()
            raise ValueError(f"2022 crosswalk dataset contains missing values:\n{missing_details_2022}")

        # Convert columns to appropriate types and clean the PUMA field for 2012 dataset
        crosswalk_2012_df['PUMA'] = crosswalk_2012_df['PUMA'].astype(str).str.strip().str.lstrip('0')
        crosswalk_2012_df['allocation factor'] = crosswalk_2012_df['allocation factor'].astype(float)

        # Convert columns to appropriate types and clean the PUMA field for 2022 dataset
        crosswalk_2022_df['PUMA'] = crosswalk_2022_df['PUMA'].astype(str).str.strip().str.lstrip('0')
        crosswalk_2022_df['allocation factor'] = crosswalk_2022_df['allocation factor'].astype(float)

        # **STEP 1: Drop Duplicate Rows on (PUMA + County_Name)**
        crosswalk_2012_df = crosswalk_2012_df.drop_duplicates(subset=['PUMA', 'County_Name'])
        crosswalk_2022_df = crosswalk_2022_df.drop_duplicates(subset=['PUMA', 'County_Name'])

        # **STEP 2: Normalize Allocation Factors (so they sum to 1 within each PUMA)**
        crosswalk_2012_df['allocation factor'] /= crosswalk_2012_df.groupby('PUMA')['allocation factor'].transform('sum')
        crosswalk_2022_df['allocation factor'] /= crosswalk_2022_df.groupby('PUMA')['allocation factor'].transform('sum')

        # New Validation: Check if allocation factors sum to 1 per PUMA
        sum_check_2012 = crosswalk_2012_df.groupby('PUMA')['allocation factor'].sum().round(6)
        sum_check_2022 = crosswalk_2022_df.groupby('PUMA')['allocation factor'].sum().round(6)

        # Find any PUMAs that are still not summing to 1
        incorrect_2012 = sum_check_2012[sum_check_2012 != 1]
        incorrect_2022 = sum_check_2022[sum_check_2022 != 1]

        if not incorrect_2012.empty:
            raise ValueError(f"2012 Crosswalk Error: Some PUMAs do not sum to 1 after normalization:\n{incorrect_2012}")

        if not incorrect_2022.empty:
            raise ValueError(f"2022 Crosswalk Error: Some PUMAs do not sum to 1 after normalization:\n{incorrect_2022}")

        logging.info("Both 2012 and 2022 crosswalk datasets loaded, cleaned, and normalized successfully.")
        return crosswalk_2012_df, crosswalk_2022_df

    except Exception as e:
        raise ValueError(f"Error loading crosswalk data: {e}")


def load_income_limits(filepath):
    """
    Load income limits data from a CSV file and perform validation checks.

    Parameters:
    filepath (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Loaded and validated income limits data.

    Raises:
    ValueError: If required columns are missing, data contains missing values, or invalid data types are found.
    """
    try:
        # Load the dataset
        income_limits_df = pd.read_csv(filepath)

        # Define required columns
        required_columns = ['County_Name'] + [f'il{threshold}_p{size}' for threshold in [30, 50, 80] for size in range(1, 9)]

        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in income_limits_df.columns]
        if missing_columns:
            raise ValueError(f"Income limits data is missing required columns: {missing_columns}")

        # Check for missing values
        missing_county_names = income_limits_df['County_Name'].isnull().sum()
        if missing_county_names > 0:
            raise ValueError(f"Income limits data contains {missing_county_names} rows with missing 'County_Name' values.")

        missing_values_summary = {
            col: income_limits_df[col].isnull().sum()
            for col in required_columns
            if income_limits_df[col].isnull().sum() > 0
        }
        if missing_values_summary:
            missing_info = "\n".join([f"{col}: {count} missing values" for col, count in missing_values_summary.items()])
            raise ValueError(f"Income limits data contains missing values in the following columns:\n{missing_info}")

        # Convert columns to numeric
        income_limit_columns = [f'il{threshold}_p{size}' for threshold in [30, 50, 80] for size in range(1, 9)]
        income_limits_df[income_limit_columns] = income_limits_df[income_limit_columns].apply(pd.to_numeric, errors='coerce')

        # Re-check for any non-numeric values
        invalid_values = income_limits_df[income_limit_columns].isnull().sum()
        if invalid_values.sum() > 0:
            invalid_columns = [col for col in income_limit_columns if income_limits_df[col].isnull().sum() > 0]
            raise ValueError(f"Income limits data contains invalid or non-numeric values in the following columns: {', '.join(invalid_columns)}")

    except Exception as e:
        raise ValueError(f"Error loading or validating income limits data from {filepath}: {e}")

    logging.info("Income limits data loaded successfully and all checks passed.")
    return income_limits_df

from pandas.api import types as pd_types
import logging

def load_incarceration_df(filepath=None):
    """
    Load incarceration data from a CSV file, coercing County_Name to string
    and validating its contents.

    Parameters
    ----------
    filepath : str or None
        Path to the incarceration CSV file. If None, returns None.

    Returns
    -------
    pd.DataFrame or None
        A DataFrame with exactly these columns:
          - State
          - County_Name         (dtype: string, never literal "0")
          - Ttl_Incarc          (int)
          - Ttl_Minority_Incarc (int)
          - Ttl_White_Incarc    (int)

    Raises
    ------
    ValueError
        If the file is missing required columns, or if County_Name
        is not textual or contains the sentinel "0".
    """
    if filepath is None:
        logging.info("No incarceration data provided.")
        return None

    # Only load the columns we need, force County_Name to pandas StringDtype
    needed_cols = [
        "State",
        "County_Name",
        "Ttl_Incarc",
        "Ttl_Minority_Incarc",
        "Ttl_White_Incarc",
    ]

    try:
        df = pd.read_csv(
            filepath,
            usecols=needed_cols,
            dtype={"County_Name": "string"},
            low_memory=False,
        )
    except Exception as e:
        raise ValueError(f"Failed to read incarceration CSV at {filepath!r}: {e}")

    # Check that all needed columns are present
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Incarceration data missing columns: {missing!r}")

    # Ensure County_Name is truly a string dtype
    if not pd_types.is_string_dtype(df["County_Name"]):
        raise ValueError(
            f"'County_Name' must be text, but got dtype {df['County_Name'].dtype}"
        )

    # Guard against the “all zeros” bug: if any entry is exactly "0", it's almost certainly wrong
    zero_mask = df["County_Name"] == "0"
    if zero_mask.any():
        bad_idxs = df.index[zero_mask][:5].tolist()
        raise ValueError(
            f"Found literal '0' in County_Name at rows {bad_idxs}; "
            "please verify the CSV path and contents."
        )

    logging.info(
        "Loaded incarceration data from %r (%d rows)",
        filepath,
        len(df)
    )
    return df

import os
import numpy as np
import logging


def load_hud_hcv_data(config: dict) -> pd.DataFrame:
    """
    Load HUD HCV data, clean negative codes/"NA" → NaN, drop "Missing County",
    and—if config['verbose'] is True—write a text report and CSV summary in config['output_directory'].
    """
    # Extract configuration
    filepath = config['hud_hcv_data_path']
    state = config['state']
    year = config['year']
    verbose = config.get('verbose', False)
    output_dir = config['output_directory']

    try:
        raw_df = pd.read_csv(filepath, dtype=str)[
            ['Name', 'Subsidized units available', '% Minority', '%White Non-Hispanic']
        ].copy()
    except Exception as e:
        raise ValueError(f"Error loading HUD HCV data from {filepath}: {e}")

    if verbose:
        os.makedirs(output_dir, exist_ok=True)

    missing_county_info = []
    neg_code_info = []
    cleaned_rows = []
    code_mapping = {
        "NA": "Not applicable",
        "-1": "Missing",
        "-4": "Suppressed ( < 11 families )",
        "-5": "Non-reporting ( < 50% reported )",
    }

    for _, row in raw_df.iterrows():
        county = row['Name'].strip()
        sub_str = (row['Subsidized units available'] or "").strip()
        min_str = (row['% Minority'] or "").strip()
        wht_str = (row['%White Non-Hispanic'] or "").strip()

        if county == "Missing County":
            missing_county_info.append((county, sub_str, min_str, wht_str))
            continue

        cleaned_vals = {}
        for col in ['Subsidized units available', '% Minority', '%White Non-Hispanic']:
            raw_val = (row[col] or "").strip()
            if raw_val in code_mapping:
                neg_code_info.append((county, col, raw_val, code_mapping[raw_val]))
                cleaned_vals[col] = np.nan
            else:
                try:
                    cleaned_vals[col] = float(raw_val.replace(",", "")) if raw_val else np.nan
                except:
                    raise ValueError(
                        f"Invalid value '{raw_val}' in '{col}' for '{county}'."
                    )

        cleaned_rows.append({
            'Name': county,
            'Subsidized units available': cleaned_vals['Subsidized units available'],
            '% Minority': cleaned_vals['% Minority'],
            '%White Non-Hispanic': cleaned_vals['%White Non-Hispanic']
        })

    hud_cleaned_df = pd.DataFrame(cleaned_rows, columns=[
        'Name', 'Subsidized units available', '% Minority', '%White Non-Hispanic'
    ])

    if verbose:
        txt_path = os.path.join(output_dir, f"{state}_{year}_hud_hcv_report.txt")
        lines = [f"{state} {year}", ""]

        if missing_county_info:
            total_missing = 0.0
            for _, sub_str, _, _ in missing_county_info:
                try:
                    total_missing += float(sub_str.replace(",", ""))
                except:
                    pass
            lines.append(
                f'There was "Missing County" data for {state} in {year}, '
                f'totaling {int(total_missing)} Subsidized Units.'
            )
            lines.append("")

        if neg_code_info:
            unique_counties = {entry[0] for entry in neg_code_info}
            count = len(unique_counties)
            lines.append(
                f"{count} county{'ies' if count > 1 else 'y'} had incomplete reporting:"
            )
            lines.append("")
            for county, col, code_str, explanation in neg_code_info:
                var_desc = {
                    'Subsidized units available': "Subsidized Units Available",
                    '% Minority': "% Minority",
                    '%White Non-Hispanic': "% White Non-Hispanic"
                }[col]
                lines.append(
                    f"{county} {state} {year}: ({code_str}) → {explanation} for '{var_desc}'."
                )
            lines.append("")

        if not missing_county_info and not neg_code_info:
            lines.append("No missing‐county rows and no negative codes found.")
            lines.append("")

        with open(txt_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        logging.info(f"Wrote HUD HCV report to: {txt_path}")

        problem_counties = {
            entry[0] for entry in neg_code_info
        }.union({entry[0] for entry in missing_county_info})

        summary_rows = []
        for county in sorted(problem_counties):
            raw_match = raw_df.loc[raw_df['Name'] == county].iloc[0]
            summary_rows.append({
                'Year': year,
                'State': state,
                'County': county,
                'Subsidized units available': raw_match['Subsidized units available'],
                '% Minority': raw_match['% Minority'],
                '%White Non-Hispanic': raw_match['%White Non-Hispanic']
            })

        summary_df = pd.DataFrame(summary_rows, columns=[
            'Year', 'State', 'County',
            'Subsidized units available', '% Minority', '%White Non-Hispanic'
        ])
        csv_path = os.path.join(output_dir, f"{state}_{year}_hud_hcv_summary.csv")
        summary_df.to_csv(csv_path, index=False)
        logging.info(f"Wrote HUD HCV summary CSV to: {csv_path}")

    logging.info("Finished loading & cleaning HUD HCV data.")
    return hud_cleaned_df
