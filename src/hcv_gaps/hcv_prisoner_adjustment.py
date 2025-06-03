"""hcv_prisoner_adjustment.py
Adjusts incarcerated persons HCV eligibility

This module contains a function to adjust Housing Choice Voucher (HCV) eligibility by removing incarcerated individuals
based on county prisoner counts and demographics. It provides options to directly identify prisoners using GQTYPE,
and to perform race-based sampling when adjusting eligibility.

Functions:

1. stratified_selection_for_incarcerated_individuals(eligibility_df, incarceration_df=None, prisoners_identified_by_GQTYPE2=False, race_sampling=False, verbose=False):
    Adjusts HCV eligibility of group-quartered individuals based on county prisoner counts and demographics. Optionally
    uses GQTYPE to identify prisoners directly or performs race-based sampling.

Usage:
To use, import this module and call the `stratified_selection_for_incarcerated_individuals` function with the eligibility
dataframe and optionally the incarceration dataframe.

Example:
    import hcv_prisoner_adjustment as hcv_prisoner

    ipums_df = load_ipums_data('path_to_ipums_data.csv')
    incarceration_df = load_incarceration_data('path_to_incarceration_data.csv')

    adjusted_ipums_df = hcv_prisoner.stratified_selection_for_incarcerated_individuals(ipums_df, incarceration_df)
"""
import logging
import numpy as np

logging.info("This is a log message from hcv_prisoner_adjustment.py")

def stratified_selection_for_incarcerated_individuals(
        eligibility_df,
        incarceration_df=None,
        prisoners_identified_by_GQTYPE2=False,
        race_sampling=False,
        verbose=False
    ):
    """
    Adjusts HCV eligibility by removing/incorporating incarcerated individuals.

    This optimized version pre-groups potential inmates by county and uses a
    vectorized random selection (shuffle + cumulative sum) to meet target weights
    in one pass instead of repeated .sample() loops.

    Parameters:
    ----------
    eligibility_df : pd.DataFrame
        IPUMS dataframe with eligibility determinations.
    incarceration_df : pd.DataFrame, optional
        DataFrame with columns ['County_Name', 'Ttl_Incarc',
        'Ttl_White_Incarc', 'Ttl_Minority_Incarc'].
    prisoners_identified_by_GQTYPE2 : bool
        If True, mark GQTYPE==2 individuals ineligible directly.
    race_sampling : bool
        If True, perform race-based sampling; else remove by total count.
    verbose : bool
        If True, logs counts removed per county/race.

    Returns:
    -------
    pd.DataFrame
        Updated eligibility DataFrame with incarcerated individuals removed.
    """
    # Columns to zero out for ineligible households
    eligibility_columns = [
        'Eligible_at_30%', 'Eligible_at_50%', 'Eligible_at_80%',
        'Weighted_Eligibility_Count_30%', 'Weighted_Eligibility_Count_50%', 'Weighted_Eligibility_Count_80%',
        'Weighted_White_HH_Eligibility_Count_30%', 'Weighted_White_HH_Eligibility_Count_50%', 'Weighted_White_HH_Eligibility_Count_80%',
        'Weighted_Minority_HH_Eligibility_Count_30%', 'Weighted_Minority_HH_Eligibility_Count_50%', 'Weighted_Minority_HH_Eligibility_Count_80%'
    ]

    # Case 1: direct identification by GQTYPE
    if prisoners_identified_by_GQTYPE2:
        prisoners = eligibility_df[eligibility_df['GQTYPE'] == 2]
        for county, grp in prisoners.groupby('County_Name'):
            eligibility_df.loc[grp.index, eligibility_columns] = 0
            if verbose:
                logging.info(f"County: {county}, Removed Prisoner Count: {grp['REALHHWT'].sum()}")
        if verbose:
            logging.info(f"Total Prisoners Marked Ineligible: {prisoners['REALHHWT'].sum()}")
        return eligibility_df

    # If no incarceration data, skip
    if incarceration_df is None:
        if verbose:
            logging.info("No incarceration data provided, skipping adjustment.")
        return eligibility_df

    # Filter potential inmates: GQTYPE 1, single-family, eligible at 80%
    potential = eligibility_df[
        (eligibility_df['GQTYPE'] == 1) &
        (eligibility_df['FAMSIZE'] == 1) &
        (eligibility_df['Eligible_at_80%'] == 1)
    ].copy()

    # Use categorical dtypes for faster grouping
    potential['County_Name'] = potential['County_Name'].astype('category')
    potential['RACE'] = potential['RACE'].astype('category')

    # Pre-group once by county
    county_groups = potential.groupby('County_Name')

    def select_inmates_vec(df, target, tol=5):
        """
        Randomly selects rows until cumulative REALHHWT ~ target using one shuffle + cumsum.
        """
        if df.empty or target <= 0:
            return []
        # shuffle via random key
        df2 = df.assign(_r=np.random.rand(len(df)))
        df2 = df2.sort_values('_r')
        csum = df2['REALHHWT'].cumsum()
        # mask A: all rows where csum <= target
        maskA = csum <= target
        # mask B: up to the row closest to target
        idx_closest = (csum - target).abs().idxmin()
        maskB = df2.index <= idx_closest
        # compare which gets closer to target
        sumA = csum[maskA].sum()
        sumB = csum[maskB].sum()
        best = maskA if abs(sumA - target) < abs(sumB - target) else maskB
        return df2.index[best]

    # Loop through incarceration data
    for _, incar in incarceration_df.iterrows():
        county = incar['County_Name']
        if verbose:
            logging.info(f"Processing county: {county}, Total Incarcerated: {incar['Ttl_Incarc']}, Race Sampling: {race_sampling}")
        # retrieve pre-grouped DataFrame
        try:
            df_cty = county_groups.get_group(county)
        except KeyError:
            if verbose:
                logging.info(f"No potential inmates for county: {county}")
            continue

        if race_sampling:
            whites    = df_cty[df_cty['RACE'] == 1]
            minorities= df_cty[df_cty['RACE'] != 1]
            sel_w = select_inmates_vec(whites,    incar['Ttl_White_Incarc'])
            sel_m = select_inmates_vec(minorities, incar['Ttl_Minority_Incarc'])
            eligibility_df.loc[sel_w, eligibility_columns] = 0
            eligibility_df.loc[sel_m, eligibility_columns] = 0
            if verbose:
                logging.info(f"County: {county}, Adjusted White REALHHWT: {eligibility_df.loc[sel_w, 'REALHHWT'].sum()}")
                logging.info(f"County: {county}, Adjusted Minority REALHHWT: {eligibility_df.loc[sel_m, 'REALHHWT'].sum()}")
        else:
            sel_all = select_inmates_vec(df_cty, incar['Ttl_Incarc'])
            eligibility_df.loc[sel_all, eligibility_columns] = 0
            if verbose:
                logging.info(f"County: {county}, Adjusted Total REALHHWT: {eligibility_df.loc[sel_all, 'REALHHWT'].sum()}")

    return eligibility_df
