"""
hcv_final_outputs.py

This module contains functions to save the final HCV eligibility DataFrame and to generate
and save a summary of eligibility counts and voucher gap calculations for each county.


Functions:
    - save_eligibility_dataframe(eligibility_df, output_dir, state, year):
          Saves the eligibility DataFrame to a CSV file using the state and year.
    - calculate_voucher_gap_and_save(ipums_eligibility_df, hud_hcv_df, output_dir, state, year, display_race_stats=False):
          Calculates the voucher gap and related statistics, then saves the summary to a CSV file.
          Optionally includes race statistics if display_race_stats is True.
"""

import os
import logging
from file_utils import (
    clean_eligibility_df,   
    tidy_summary_df,        
)

logging.info("Loaded hcv_final_outputs module")

def save_eligibility_dataframe(eligibility_df, output_dir, state, year):
    """
    Tidies and saves the eligibility DataFrame to a CSV file named with the state and year.
    If the state is CT and the YEAR is 2023, all County Names are updated to CT's new (as of 2023) Planning Regions'
    
    Parameters:
        eligibility_df (pd.DataFrame): The processed IPUMS DataFrame with eligibility data.
        output_dir (str): The directory where the output file should be saved.
        state (str): The state abbreviation (e.g., "FL").
        year (int or str): The year (e.g., 2022).
    
    Returns:
        None
    """
    
    # Clean the Eligibility df--Removes temporary columns.
    eligibility_df = clean_eligibility_df(eligibility_df, state, year)
    
    file_name = f"{state}_{year}_eligibility.csv"
    file_path = os.path.join(output_dir, file_name)
    try:
        eligibility_df.to_csv(file_path, index=False)
        logging.info(f"Eligibility DataFrame saved successfully to {file_path}")
    except Exception as e:
        logging.error(f"Error saving eligibility DataFrame: {e}")

def calculate_voucher_gap_and_save(ipums_eligibility_df, hud_hcv_df, output_dir, state, year, display_race_stats=False):
    """
    Calculates the voucher gap and HCV allocation rate for each county, including:
      - Overall eligibility counts.
      - Renter-only aggregates and corresponding gap calculations.
      - Optionally, race-specific statistics.
    The final summary is saved to a CSV file named with the provided state and year.
    The eligibility DataFrame is also saved in the same directory.
    
    Parameters:
        ipums_eligibility_df (pd.DataFrame): The IPUMS eligibility DataFrame.
        hud_hcv_df (pd.DataFrame): The HUD HCV DataFrame.
        output_dir (str): The directory where the output files should be saved.
        state (str): The state abbreviation (e.g., "FL").
        year (int or str): The year (e.g., 2022).
        display_race_stats (bool, optional): If True, includes race statistics in the output (default is False).
    
    Returns:
        pd.DataFrame: The summary DataFrame with calculated voucher gaps and related statistics.
    """
    def add_race_stats(df, hud_hcv):
        # Merge with HUD HCV data to get race stats.
        df = df.merge(
            hud_hcv[['Name', '% Minority', '%White Non-Hispanic']],
            left_on='County_Name', right_on='Name', how='left'
        )
        # Add weighted counts for minorities and whites (for all eligible households).
        df['Weighted_Minority_Count_30%'] = ipums_eligibility_df.groupby('County_Name')['Weighted_Minority_HH_Eligibility_Count_30%'].sum().values
        df['Weighted_White_Count_30%'] = ipums_eligibility_df.groupby('County_Name')['Weighted_White_HH_Eligibility_Count_30%'].sum().values

        df['Weighted_Minority_Count_50%'] = ipums_eligibility_df.groupby('County_Name')['Weighted_Minority_HH_Eligibility_Count_50%'].sum().values
        df['Weighted_White_Count_50%'] = ipums_eligibility_df.groupby('County_Name')['Weighted_White_HH_Eligibility_Count_50%'].sum().values

        df['Weighted_Minority_Count_80%'] = ipums_eligibility_df.groupby('County_Name')['Weighted_Minority_HH_Eligibility_Count_80%'].sum().values
        df['Weighted_White_Count_80%'] = ipums_eligibility_df.groupby('County_Name')['Weighted_White_HH_Eligibility_Count_80%'].sum().values

        # Calculate percentages.
        df['% Eligible Minority at 30%'] = (df['Weighted_Minority_Count_30%'] / df['Weighted_Eligibility_Count_30%']) * 100
        df['% Eligible White at 30%'] = (df['Weighted_White_Count_30%'] / df['Weighted_Eligibility_Count_30%']) * 100

        df['% Eligible Minority at 50%'] = (df['Weighted_Minority_Count_50%'] / df['Weighted_Eligibility_Count_50%']) * 100
        df['% Eligible White at 50%'] = (df['Weighted_White_Count_50%'] / df['Weighted_Eligibility_Count_50%']) * 100

        df['% Eligible Minority at 80%'] = (df['Weighted_Minority_Count_80%'] / df['Weighted_Eligibility_Count_80%']) * 100
        df['% Eligible White at 80%'] = (df['Weighted_White_Count_80%'] / df['Weighted_Eligibility_Count_80%']) * 100

        return df

    # Summarize overall eligibility data by county.
    summary_df = ipums_eligibility_df.groupby('County_Name').agg({
        'Weighted_Eligibility_Count_30%': 'sum',
        'Weighted_Eligibility_Count_50%': 'sum',
        'Weighted_Eligibility_Count_80%': 'sum'
    }).reset_index()
    
    # Compute renter-only aggregates.
    renters_df = ipums_eligibility_df[ipums_eligibility_df['OWNERSHP'] == 2]

    renter_elig = renters_df.groupby('County_Name').agg({
        'Weighted_Eligibility_Count_30%': 'sum',
        'Weighted_Eligibility_Count_50%': 'sum',
        'Weighted_Eligibility_Count_80%': 'sum'
    }).rename(columns={
        'Weighted_Eligibility_Count_30%': 'Weighted_Renter_Eligibility_Count_30%',
        'Weighted_Eligibility_Count_50%': 'Weighted_Renter_Eligibility_Count_50%',
        'Weighted_Eligibility_Count_80%': 'Weighted_Renter_Eligibility_Count_80%'
    }).reset_index()

    renter_minority = renters_df.groupby('County_Name').agg({
        'Weighted_Minority_HH_Eligibility_Count_30%': 'sum',
        'Weighted_Minority_HH_Eligibility_Count_50%': 'sum',
        'Weighted_Minority_HH_Eligibility_Count_80%': 'sum'
    }).rename(columns={
        'Weighted_Minority_HH_Eligibility_Count_30%': 'Weighted_Renter_Minority_Count_30%',
        'Weighted_Minority_HH_Eligibility_Count_50%': 'Weighted_Renter_Minority_Count_50%',
        'Weighted_Minority_HH_Eligibility_Count_80%': 'Weighted_Renter_Minority_Count_80%'
    }).reset_index()

    renter_white = renters_df.groupby('County_Name').agg({
        'Weighted_White_HH_Eligibility_Count_30%': 'sum',
        'Weighted_White_HH_Eligibility_Count_50%': 'sum',
        'Weighted_White_HH_Eligibility_Count_80%': 'sum'
    }).rename(columns={
        'Weighted_White_HH_Eligibility_Count_30%': 'Weighted_Renter_White_Count_30%',
        'Weighted_White_HH_Eligibility_Count_50%': 'Weighted_Renter_White_Count_50%',
        'Weighted_White_HH_Eligibility_Count_80%': 'Weighted_Renter_White_Count_80%'
    }).reset_index()

    # Merge the renter-only aggregates into summary_df.
    summary_df = summary_df.merge(renter_elig, on='County_Name', how='left')
    summary_df = summary_df.merge(renter_minority, on='County_Name', how='left')
    summary_df = summary_df.merge(renter_white, on='County_Name', how='left')

    # Calculate renter percentages.
    summary_df['% Eligible Minority (Renters) at 30%'] = (summary_df['Weighted_Renter_Minority_Count_30%'] / summary_df['Weighted_Renter_Eligibility_Count_30%']) * 100
    summary_df['% Eligible White (Renters) at 30%'] = (summary_df['Weighted_Renter_White_Count_30%'] / summary_df['Weighted_Renter_Eligibility_Count_30%']) * 100

    summary_df['% Eligible Minority (Renters) at 50%'] = (summary_df['Weighted_Renter_Minority_Count_50%'] / summary_df['Weighted_Renter_Eligibility_Count_50%']) * 100
    summary_df['% Eligible White (Renters) at 50%'] = (summary_df['Weighted_Renter_White_Count_50%'] / summary_df['Weighted_Renter_Eligibility_Count_50%']) * 100

    summary_df['% Eligible Minority (Renters) at 80%'] = (summary_df['Weighted_Renter_Minority_Count_80%'] / summary_df['Weighted_Renter_Eligibility_Count_80%']) * 100
    summary_df['% Eligible White (Renters) at 80%'] = (summary_df['Weighted_Renter_White_Count_80%'] / summary_df['Weighted_Renter_Eligibility_Count_80%']) * 100

    # Normalize county names for merging with HUD data.
    summary_df['County_Name_Normalized'] = summary_df['County_Name'].str.lower().str.strip().str.replace(r"[.'’]", "", regex=True)
    hud_hcv_df['Name_Normalized'] = hud_hcv_df['Name'].str.lower().str.strip().str.replace(r"[.'’]", "", regex=True)

    # Merge with HUD data.
    summary_df = summary_df.merge(
        hud_hcv_df[['Name_Normalized', 'Subsidized units available', '% Minority']],
        left_on='County_Name_Normalized', right_on='Name_Normalized', how='left'
    )

    # Drop temporary normalized columns.
    summary_df.drop(columns=['Name_Normalized', 'County_Name_Normalized'], inplace=True)

    # Calculate overall voucher gaps.
    summary_df['Voucher_Gap_30%'] = summary_df['Weighted_Eligibility_Count_30%'] - summary_df['Subsidized units available']
    summary_df['Voucher_Gap_50%'] = summary_df['Weighted_Eligibility_Count_50%'] - summary_df['Subsidized units available']
    summary_df['Voucher_Gap_80%'] = summary_df['Weighted_Eligibility_Count_80%'] - summary_df['Subsidized units available']

    # Calculate renter-specific voucher gaps.
    summary_df['Renter_Voucher_Gap_30%'] = summary_df['Weighted_Renter_Eligibility_Count_30%'] - summary_df['Subsidized units available']
    summary_df['Renter_Voucher_Gap_50%'] = summary_df['Weighted_Renter_Eligibility_Count_50%'] - summary_df['Subsidized units available']
    summary_df['Renter_Voucher_Gap_80%'] = summary_df['Weighted_Renter_Eligibility_Count_80%'] - summary_df['Subsidized units available']

    # Calculate HCV allocation rates.
    summary_df['HCV_Allocation_Rate_30%'] = (summary_df['Subsidized units available'] / summary_df['Weighted_Eligibility_Count_30%']) * 100
    summary_df['HCV_Allocation_Rate_50%'] = (summary_df['Subsidized units available'] / summary_df['Weighted_Eligibility_Count_50%']) * 100
    summary_df['HCV_Allocation_Rate_80%'] = (summary_df['Subsidized units available'] / summary_df['Weighted_Eligibility_Count_80%']) * 100

    # Calculate minority subsidy recipients.
    summary_df['Minority_Subsidy_Recipients'] = summary_df['Subsidized units available'] * (summary_df['% Minority'] / 100)

    # Optionally add detailed race stats.
    if display_race_stats:
        summary_df = add_race_stats(summary_df, hud_hcv_df)

    summary_df = tidy_summary_df(summary_df, state)          

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info("Created missing output directory: " + output_dir)

    suffix = "_with_race_stats" if display_race_stats else ""
    summary_path = os.path.join(
        output_dir, f"{state}_{year}_HCV_Gap_Summary_Table{suffix}.csv"
    )
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"HCV Gap Summary DataFrame saved successfully to {summary_path}")

    # Save the cleaned eligibility DataFrame
    save_eligibility_dataframe(ipums_eligibility_df, output_dir, state, year)

    logging.info(f"Finished processing voucher gap for {state} in {year}")
    return summary_df

