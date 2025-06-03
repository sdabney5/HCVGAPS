"""
hcv_family_feature_engineering.py
Engineers family/household level features in the IPUMS dataset.

This module contains functions for engineering family-level features and condensing households
into single rows within the IPUMS dataset.

Functions:

1. family_feature_engineering(df):
    Adds family-level features to the dataset, including education metrics and race binary columns.

2. flatten_households_to_single_rows(df):
    Condenses the DataFrame to one representative row per family, aggregating specified columns and
    notifying the user of any extra columns.

Usage:
To use, import this module and call the functions with the IPUMS dataframe as the argument.
The functions will return the modified DataFrame with engineered family features and condensed households.

Example:
    import hcv_family_feature_engineering as hcv_engineering

    ipums_df = hcv_engineering.family_feature_engineering(ipums_df)
    ipums_df = hcv_engineering.flatten_households_to_single_rows(ipums_df)
"""

#imports
import pandas as pd
import logging
logging.info("This is a log message from hcv_family_feature_engineering")

def family_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer household–level features in an IPUMS-style person-level DataFrame.

    The function collapses individual-level records (one row per person) into
    household-level indicators, then merges those indicators back onto the
    original DataFrame so that every person row “knows” its household’s
    characteristics.

    **Key traits computed**

    ─ Household composition
        • SENIOR_HOUSEHOLD   – 1 if anyone is 65+  
        • NUM_CHILDREN       – count of members < 18  
        • MARRIED_FAMILY_HH  – 1 if `HHTYPE == 1` for any member  
        • SINGLE_PARENT_HH   – 1 if `HHTYPE in {2, 3}` for any member  

    ─ Veteran / employment
        • VET_HOUSEHOLD      – 1 if any member has `VETSTAT == 2`  
        • EMPLOYED           – 1 if any member has `EMPSTAT == 1`  

    ─ Education flags (highest attained by any member)
        • HS_COMPLETE        – 1 if anyone has `EDUCD ≥ 62`  
        • BACHELOR_COMPLETE  – 1 if anyone has `EDUCD == 101`  
        • GRAD_SCHOOL        – 1 if anyone has `EDUCD > 101`  

    ─ Race (one-hot, mutually exclusive; exactly one “1” per household)
        • White_HH, Black_HH, Asian_HH, Mixed_Race_HH, Other_Race_HH  
          Race is taken from the head of household (`RELATE == 1`);
          if absent, the spouse (`RELATE == 2`); otherwise the first member.

    **Performance notes**

    * Vectorised masks → one C-level `groupby`, no Python lambdas.
    * Memory overhead is a handful of temporary `_` columns that are removed
      before return.
    * Omits the legacy string column `HOUSEHOLD_RACE` and the flag
      `TWO_COLLEGE_GRADS`, which downstream code no longer uses.

    Parameters
    ----------
    df : pandas.DataFrame
        Person-level IPUMS data containing at least the columns  
        `FAMILYNUMBER, AGE, VETSTAT, EDUCD, HHTYPE, EMPSTAT, RELATE, RACE`.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with the household-level columns listed above
        merged on.  Column names and dtypes are preserved for all
        pre-existing fields.

    Examples
    --------
    >>> people = pd.read_parquet("fl_2022_ipums.parquet")
    >>> people = family_feature_engineering(people)
    >>> people.filter(regex="_HH$|NUM_CHILDREN").head()
    """
   

    # ---------- 1. per-person Boolean / numeric flags --------------
    df["_IS_SENIOR"]      = (df["AGE"]  > 64).astype("uint8")
    df["_IS_CHILD"]       = (df["AGE"]  < 18).astype("uint8")
    df["_IS_VET"]         = (df["VETSTAT"] == 2).astype("uint8")
    df["_EDU_HS_PLUS"]    = (df["EDUCD"] >= 62).astype("uint8")
    df["_EDU_BACHELOR"]   = (df["EDUCD"] == 101).astype("uint8")
    df["_EDU_GRAD"]       = (df["EDUCD"] > 101).astype("uint8")
    df["_HHTYPE_MARRIED"] = (df["HHTYPE"] == 1).astype("uint8")
    df["_HHTYPE_SINGLE"]  = ((df["HHTYPE"] == 2) | (df["HHTYPE"] == 3)).astype("uint8")
    df["_EMPLOYED"]       = (df["EMPSTAT"] == 1).astype("uint8")

    # ---------- 2. pick representative member & race dummies -------
    priority = df["RELATE"].replace({1: 0, 2: 1}).fillna(2)
    rep_idx  = priority.groupby(df["FAMILYNUMBER"], sort=False).idxmin()

    rep = df.loc[rep_idx, ["FAMILYNUMBER", "RACE"]].set_index("FAMILYNUMBER")
    rep["White_HH"]       =  (rep["RACE"] == 1).astype("uint8")
    rep["Black_HH"]       =  (rep["RACE"] == 2).astype("uint8")
    rep["Asian_HH"]       =   rep["RACE"].isin([4, 5, 6]).astype("uint8")
    rep["Mixed_Race_HH"]  =   rep["RACE"].isin([7, 8, 9]).astype("uint8")
    rep["Other_Race_HH"]  = (
        1 - rep[["White_HH", "Black_HH", "Asian_HH", "Mixed_Race_HH"]].sum(axis=1)
    ).astype("uint8")
    rep = rep.drop(columns="RACE")

    # ---------- 3. household-level aggregation ---------------------
    fam = (
        df.groupby("FAMILYNUMBER", sort=False)
          .agg(
              SENIOR_HOUSEHOLD  = ("_IS_SENIOR",      "max"),
              NUM_CHILDREN      = ("_IS_CHILD",       "sum"),
              VET_HOUSEHOLD     = ("_IS_VET",         "max"),
              HS_COMPLETE       = ("_EDU_HS_PLUS",    "max"),
              BACHELOR_COMPLETE = ("_EDU_BACHELOR",   "max"),
              GRAD_SCHOOL       = ("_EDU_GRAD",       "max"),
              MARRIED_FAMILY_HH = ("_HHTYPE_MARRIED", "max"),
              SINGLE_PARENT_HH  = ("_HHTYPE_SINGLE",  "max"),
              EMPLOYED          = ("_EMPLOYED",       "max"),
          )
          .astype({
              "SENIOR_HOUSEHOLD":"uint8",
              "NUM_CHILDREN"    :"uint16",
              "VET_HOUSEHOLD"   :"uint8",
              "HS_COMPLETE"     :"uint8",
              "BACHELOR_COMPLETE":"uint8",
              "GRAD_SCHOOL"     :"uint8",
              "MARRIED_FAMILY_HH":"uint8",
              "SINGLE_PARENT_HH":"uint8",
              "EMPLOYED"        :"uint8"
          })
    )

    # ---------- 4. attach race dummy columns -----------------------
    fam = fam.join(rep, how="left")

    # ---------- 5. merge back to person-level DataFrame ------------
    logging.info("Merging aggregated family features back to main DataFrame")
    df = df.merge(fam.reset_index(), on="FAMILYNUMBER", how="left")

    # ---------- 6. clean temporary helper columns ------------------
    df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True)

    return df



#Function to condense families/households to a single row
def flatten_households_to_single_rows(df):
    """
    Condense the dataframe so there's only one representative row per family.

    This function condenses the DataFrame so there's only one representative row per family. It drops specified
    columns and retains unspecified columns. It aggregates certain columns by taking the first value for each
    family, sums specified columns, and retains the first value for unspecified columns. If the DataFrame contains
    additional columns not specified in the function, it takes the first value for these columns and notifies the user.

    Parameters:
    ----------
    df : pd.DataFrame
        The input dataframe.

    Returns:
    -------
    pd.DataFrame
        The condensed dataframe with one row per family.

    Notes:
    -----
    - The function drops columns that are not needed for the analysis.
    - The function aggregates certain columns by taking the first value for each family.
    - For specified columns that require summation, the function aggregates them by summing.
    - For columns that are not specifically listed, the function retains them and applies the 'first' aggregation.
    - If the DataFrame contains additional columns not specified in the function, the function will take the first value
      for these columns and notify the user.

    Verification:
    -------------
    Verified on 2024-06-25 with:
    - test_data_1
    - production_data
    """
    # Columns to drop
    cols_to_drop = ['OTHERINCOME_FAMILY', 'OTHERINCOME_PERSONAL', 'HHINCOME', 'QOWNERSH', 'QRENTGRS', 'QHHINCOME',
                    'PERNUM', 'PERWT', 'FAMUNIT', 'AGE', 'MARST', 'BIRTHYR', 'HCOVANY', 'SCHOOL', 'EDUC', 'EDUCD',
                    'GRADEATT', 'GRADEATTD', 'SCHLTYPE', 'RELATE', 'RELATED', 'GCRESPON', 'QAGE', 'QMARST',
                    'QSEX', 'QHINSEMP', 'QHINSPUR', 'QHINSTRI', 'QHINSCAI', 'QHINSCAR', 'QHINSVA', 'QHINSIHS',
                    'QEDUC', 'QGRADEAT', 'QSCHOOL', 'QMIGRAT1', 'QMOVEDIN', 'QVETSTAT', 'QTRANTIM', 'QGCRESPO',
                    'INCWAGE', 'INCSS', 'INCWELFR', 'INCINVST', 'INCRETIR', 'INCSUPP', 'INCOTHER', 'INCEARN',
                    'VETSTAT', 'VETSTATD', 'FTOTINC']

    # Drop the columns
    logging.info('Flattening households to Single Row...')
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # The columns where we'll just take the first value since they should all be the same for a family
    cols_first_value = ['County_Name', 'YEAR', 'HHWT', 'MULTYEAR', 'SAMPLE', 'SERIAL', 'CBSERIAL', 'CBHHTYPE', 'CLUSTER',
                        'REGION', 'STATEICP', 'COUNTYICP', 'COUNTYFIP', 'METRO', 'MET2013', 'MET2013ERR',
                        'CITY', 'PUMA', 'STRATA', 'GQ', 'GQTYPE', 'GQTYPED', 'OWNERSHP', 'OWNERSHPD',
                        'MORTGAGE', 'MORTAMT1', 'RENTGRS', 'FAMILYNUMBER', 'ACTUAL_HH_INCOME', 'NFAMS_B4_SPLIT',
                        'HHTYPE', 'FOODSTMP', 'VEHICLES', 'NMOTHERS', 'NFATHERS', 'MULTGEN',
                        'MULTGEND', 'FAMSIZE', 'POVERTY', 'SENIOR_HOUSEHOLD', 'NUM_CHILDREN', 'VET_HOUSEHOLD',
                        'MARRIED_FAMILY_HH', 'SINGLE_PARENT_HH', 'REALHHWT', 'EMPLOYED', 'CITIZEN',
                        'HISPAN', 'HISPAND', 'RACE', 'RACED', 'White', 'Black', 'Asian', 'Mixed Race',
                        'Other_Race', 'HS_COMPLETE', 'BACHELOR_COMPLETE',
                        'GRAD_SCHOOL', 'SEX', 'MCDC_PUMA_COUNTY_NAMES', 'Multi_County_Flag',
                        'Black_HH', 'MARRIED_FAMILY_HH', 'Asian_HH', 'Mixed_Race_HH', 'EMPSTAT', 'Other_Race_HH',
                        'SINGLE_PARENT_HH', 'White_HH']

    # Columns where we'll take the sum value
    cols_sum_value = ['UHRSWORK']

    # Group by FAMILYNUMBER and aggregate
    aggregation = {col: 'first' for col in df.columns.intersection(cols_first_value)}
    aggregation.update({col: 'sum' for col in df.columns.intersection(cols_sum_value)})

    # Capture extra columns not specified
    all_columns = set(df.columns)
    specified_columns = set(df.columns.intersection(cols_first_value)) | set(df.columns.intersection(cols_sum_value))
    extra_columns = all_columns - specified_columns

    # Add extra columns to the aggregation dictionary with 'first'
    for col in extra_columns:
        aggregation[col] = 'first'

    # Print message if there are extra columns
    if extra_columns:
        logging.info("FYI: The following extra columns were found and their first values were taken for each household:")
        logging.info(", ".join(extra_columns))

    # Group, aggregate, then put FAMILYNUMBER back as a column and reset the index
    condensed_df = df.groupby('FAMILYNUMBER').agg(aggregation)
    condensed_df['FAMILYNUMBER'] = condensed_df.index
    condensed_df.reset_index(drop=True, inplace=True)
        
    # **Ensure each FAMILYNUMBER is unique after flattening**
    assert condensed_df['FAMILYNUMBER'].is_unique, "Error: Duplicate FAMILYNUMBER values exist after flattening!"

    return condensed_df