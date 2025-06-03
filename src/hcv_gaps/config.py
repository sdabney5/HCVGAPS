"""
Configuration file for HCV analysis scripts.

This file contains paths to the CSV files and other configuration options
used in the HCV analysis process. It also supports processing multiple states by including:
- A list of state abbreviations (users can modify as needed).
- A base directory for state-specific data (relative to the repository root).
- Template strings for state-specific file paths.
- Optional API settings for fetching data instead of using local CSV files.

For IPUMS data:
  - If "ipums_data_path" is set to "API" (or left empty), the system will fetch the data via the API.
  - In that case, ensure that "api_settings" has "use_ipums_api" set to True, along with a valid API token.
  - The API call will download the data into the directory specified by "api_settings.download_dir" (default is "api_downloads").

Attributes:
    CONFIG (dict): A dictionary containing configuration options, including file paths
                   and various processing options.
      - ipums_data_path: str, A placeholder for IPUMS data. Set to "API" or empty to trigger API download.
      - crosswalk_2012_path: str, Path to the 2012 MCDC crosswalk dataset CSV file.
      - crosswalk_2022_path: str, Path to the 2022 MCDC crosswalk dataset CSV file.
      - income_limits_path: str, Path to the income limits dataset CSV file.
      - incarceration_data_path: str, Path to the incarceration dataset CSV file.
      - hud_hcv_data_path: str, Path to the HUD Picture of Subsidized Housing CSV file.
      - output_directory: str, The base directory where output files will be saved.
      - verbose: bool, If True, enables logging at INFO level.
      - prisoners_identified_by_GQTYPE2: bool, If True, uses GQTYPE == 2 for identifying prisoners.
      - race_sampling: bool, If True, performs race-based sampling when adjusting eligibility.
      - display_race_stats: bool, If True, includes race statistics in the output.
      - states: list, A list of state abbreviations to process.
      - ipums_years: list, A list of years for which to process IPUMS data.
      - data_dir: str, Base directory for state-specific data.
      - [Template strings]: Templates for state-specific file paths that will be used to update
        the core keys dynamically.
      - api_settings: dict, Optional API settings for data sources.
          Example keys:
              use_ipums_api: bool, whether to fetch IPUMS data via an API.
              ipums_api_token: str, the API token for IPUMS.
              download_dir: str, the directory where API downloads should be saved (default "api_downloads").
              use_hud_api: bool, whether to fetch HUD data via an API.
              hud_api_token: str, the API token for HUD data access.

Usage:
    To use this configuration, import this file in your script and access the CONFIG dictionary.
    Update the file paths and API settings according to your system.

Example:
    For state "ak" (Alaska), the expected files are:
      - ak_geocorr_puma_2012.csv
      - ak_geocorr_puma_2022.csv
      - ak_hud_income_limits_2022.csv
      - ak_incarc_data.csv
      - ak_hud_hcv_picsubhhds_2022.csv

The core keys (e.g., "crosswalk_2012_path") will be updated dynamically using the template strings.
"""

from .file_utils import get_default_output_directory

CONFIG = {
    # IPUMS data: if set to "API" (or left empty), the system will fetch data via the API.
    "ipums_data_path": "API",

    # The default output directory is set via file_utils. (Users can override this.)
    "output_directory": get_default_output_directory(),

    # User-entered settings:
    "states": ["fl"],
    "ipums_years": [2022],

    # Base directory for state-specific data (relative to the repository root)
    "data_dir": "data",

    # Template strings for state-specific file paths.
    "crosswalk_2012_template": "{data_dir}/{state}/{state}_geocorr_puma_2012.csv",
    "crosswalk_2022_template": "{data_dir}/{state}/{state}_geocorr_puma_2022.csv",
    "income_limits_template": "{data_dir}/{state}/{state}_hud_income_limits_2022.csv",
    "incarceration_template": "{data_dir}/{state}/{state}_incarc_data.csv",
    "hud_hcv_template": "{data_dir}/{state}/{state}_hud_hcv_picsubhhds_{year}.csv",

    # Other configuration options:
    "verbose": True,
    "prisoners_identified_by_GQTYPE2": False,
    "race_sampling": True,
    "display_race_stats": True,

    # Optional API settings for data sources:
    "api_settings": {
        "use_ipums_api": True,
        "ipums_api_token": "59cba10d8a5da536fc06b59d00bd2ef7132749b49eb77e67e1286a95",
        "download_dir": "data/api_downloads",  # API downloads will be saved in a subfolder in the data directory.
        "clear_api_cache": True,
        "use_hud_api": False, # Currently not supported, but relevant files are stored in data
        "hud_api_token": "" #Currently not supported.
    },
}
