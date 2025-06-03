"""
File Utilities Module.

This module provides functions for creating output directories.
Specifically, it creates a root output directory (e.g., on the user's Desktop),
and for each state and year combination, it creates:
  - A state folder
  - Within that state folder, a subfolder named with the state and year (e.g., FL_2019)
"""

import os
import logging
import pandas as pd
import shutil
from pathlib import Path

def create_output_structure(root_output, state, year):
    """
    Creates the complete output directory structure for a given state and year.

    The structure will be:
      root_output/                (e.g., C:/Users/<user>/Desktop/HCV_GAPS_output)
          STATE/                 (e.g., FL)
              STATE_YEAR/        (e.g., FL_2019)

    Parameters:
        root_output (str): The base output directory path.
        state (str): The state abbreviation.
        year (int or str): The year.

    Returns:
        str: The full path to the final directory created for the state and year.
    """
    # Check if root output directory already exists.
    if not os.path.exists(root_output):
        os.makedirs(root_output)
        logging.info("Created root output directory: " + root_output)

    # Create the state folder (using uppercase for consistency)
    state_dir = os.path.join(root_output, state.upper())
    if not os.path.exists(state_dir):
        os.makedirs(state_dir)
        logging.info("Created state directory: " + state_dir)

    # Create the state_year folder (e.g., FL_2019)
    final_dir = os.path.join(state_dir, "{}_{}".format(state.upper(), year))
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)
        logging.info("Created output directory for {} {}: {}".format(state.upper(), year, final_dir))

    return final_dir

def get_default_output_directory():
    """
    Determines the user's Desktop and returns a default output directory path.

    Returns:
        str: The default output directory path on the user's Desktop.
    """
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    default_output = os.path.join(desktop, "HCV_GAPS_output")
    return default_output



def clean_eligibility_df(
    elig_df: pd.DataFrame,
    state: str,
    year: int | str,
) -> pd.DataFrame:
    """
    • Drop 'State abbr.' column if present  
    • If state == 'CT' and year == 2023, map PUMA→County_Name
    """
    df = elig_df.drop(columns=["State abbr."], errors="ignore").copy()

    if state == "CT" and int(year) == 2023:
        ct_puma_to_county = {
            "20100": "Northwest Hills Planning Region CT",
            "20201": "Capitol Planning Region CT",
            "20202": "Lower Connecticut River Valley Planning Region CT",
            "20203": "Capitol Planning Region CT",
            "20204": "Capitol Planning Region CT",
            "20205": "Capitol Planning Region CT",
            "20206": "Capitol Planning Region CT",
            "20207": "Capitol Planning Region CT",
            "20301": "Northeastern Connecticut Planning Region CT",
            "20401": "Southeastern Connecticut Planning Region CT",
            "20402": "Southeastern Connecticut Planning Region CT",
            "20500": "Lower Connecticut River Valley Planning Region CT",
            "20601": "South Central Connecticut Planning Region CT",
            "20602": "South Central Connecticut Planning Region CT",
            "20603": "South Central Connecticut Planning Region CT",
            "20604": "South Central Connecticut Planning Region CT",
            "20701": "Naugatuck Valley Planning Region CT",
            "20702": "Naugatuck Valley Planning Region CT",
            "20703": "Naugatuck Valley Planning Region CT",
            "20801": "Greater Bridgeport Planning Region CT",
            "20802": "Greater Bridgeport Planning Region CT",
            "20901": "Western Connecticut Planning Region CT",
            "20902": "Western Connecticut Planning Region CT",
            "20903": "Western Connecticut Planning Region CT",
            "20904": "Western Connecticut Planning Region CT"
        }

        if "PUMA" in df.columns:
            df["County_Name"] = df.apply(
                lambda r: ct_puma_to_county.get(str(r["PUMA"]), r["County_Name"]),
                axis=1,
            )
    return df


# --------------------------------------------------------------------------- #
# 1B.  Summary-DF tidy-up  (runs AFTER all aggregations)                      #
# --------------------------------------------------------------------------- #
def tidy_summary_df(
    summary_df: pd.DataFrame,
    state: str,
) -> pd.DataFrame:
    """
    • Drop 'Name' + '% Minority_x'  
    • Rename '% Minority_y' → '% Minority'  
    • For AK & LA only, strip trailing ' County' from County_Name
    """
    df = summary_df.drop(columns=["Name", "% Minority_x"], errors="ignore").rename(
        columns={"% Minority_y": "% Minority"}, errors="ignore"
    )

    if state in {"AK", "LA"} and "County_Name" in df.columns:
        df["County_Name"] = df["County_Name"].str.replace(
            r"\s+County$", "", regex=True
        )
    return df




def clear_api_downloads(folder: str | Path) -> None:
    """
    Remove all files/subfolders inside api_downloads folder, but leave the folder itself.
    """
    folder = Path(folder)
    if not folder.exists():
        return

    for item in folder.iterdir():
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except Exception as exc:            # pylint: disable=broad-except
            logging.warning("Could not delete %s: %s", item, exc)


