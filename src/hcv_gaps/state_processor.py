"""
State Processor Module for HCV Eligibility Analysis.

This module processes multiple states and years using the global configuration.
For each state-year combination, it:
  1. Updates the configuration with state-specific file paths using template strings (except for HUD HCV, which is year-specific).
  2. Sets the current state and year in the configuration.
  3. Creates the output directory structure for final analysis outputs.
  4. Checks for and/or downloads the IPUMS data file into a dedicated subdirectory within the data directory.
  5. Updates the configuration's "ipums_data_path" with the downloaded file path.
  6. Updates the HUD HCV data path using the current year.
  7. Calls the core processing function (process_hcv_eligibility) with the updated configuration.
"""

import os
import copy
import logging
from .hcv_processing import process_hcv_eligibility
from .file_utils import create_output_structure
from .api_calls import fetch_ipums_data_api

def update_config_for_state(config, state):
    """
    Update the configuration dictionary for a specific state.
    
    This function takes the global configuration and a state abbreviation,
    and returns a new configuration dictionary with state-specific file paths updated
    using the template strings defined in the global config. (The HUD HCV data path
    is not updated here because it is year-specific and will be updated later.)
    
    Parameters:
        config (dict): The global configuration dictionary.
        state (str): The state abbreviation (expected to match directory names).
        
    Returns:
        dict: A new configuration dictionary with updated file paths for the given state.
    """
    state_config = copy.deepcopy(config)
    state_config["crosswalk_2012_path"] = config["crosswalk_2012_template"].format(
        data_dir=config["data_dir"], state=state)
    state_config["crosswalk_2022_path"] = config["crosswalk_2022_template"].format(
        data_dir=config["data_dir"], state=state)
    state_config["income_limits_path"] = config["income_limits_template"].format(
        data_dir=config["data_dir"], state=state)
    state_config["incarceration_data_path"] = config["incarceration_template"].format(
        data_dir=config["data_dir"], state=state)
    # Do not update hud_hcv_data_path here because it is year-specific.
    return state_config



def get_ipums_data_file(config):
    """
    1) If api_settings.use_ipums_api is False AND ipums_data_path is a real file, return it.
    2) If api_settings.use_ipums_api is True, fetch via IPUMS API (saving to the usual folder).
    3) If ipums_data_path is literally "API" or empty, fetch via API.
    4) Else: user has disabled API but given a bad path → error.
    """
    # unpack
    local_path = config.get("ipums_data_path", "").strip()
    use_api    = config.get("api_settings", {}).get("use_ipums_api", False)

    # 1. user wants local only
    if not use_api and local_path and local_path.upper() != "API":
        if os.path.exists(local_path):
            logging.info(f"Loading IPUMS data from local file: {local_path}")
            return local_path
        else:
            raise FileNotFoundError(
                f"CONFIG: use_ipums_api=False but ipums_data_path='{local_path}' not found."
            )

    # define where API should drop it
    base_dir = config["data_dir"]
    dl_dir   = os.path.join(base_dir, config["state"].lower(), "api_downloads", "ipums_api_downloads")
    os.makedirs(dl_dir, exist_ok=True)
    fn       = f"{config['state'].lower()}_ipums_{config['year']}.csv"
    out_path = os.path.join(dl_dir, fn)

    # 2. or 3. fetch via API
    if use_api or not local_path or local_path.upper() == "API":
        logging.info("Fetching IPUMS via API…")
        # temporarily override download_dir so any internal code that looks there will see our dl_dir
        config["api_settings"]["download_dir"] = dl_dir
        df = fetch_ipums_data_api(config)
        if df is None:
            raise RuntimeError("Failed to fetch IPUMS data from API.")
        df.to_csv(out_path, index=False)
        logging.info(f"Saved IPUMS CSV to {out_path}")
        return out_path

    # should never get here
    raise RuntimeError("IPUMS configuration logic fell through unexpectedly.")
    
    
    
def process_all_states(config):
    """
    Process HCV eligibility data for all states and years defined in the configuration.

    This function loops over each state in config["states"] and each year in config["ipums_years"],
    performing the following steps for each state-year combination:
      1. Update the configuration with state-specific file paths using template strings.
      2. Set the current state and year in the configuration.
      3. Create the state-year specific output directory.
      4. Update the HUD HCV data path with the current year.
      5. Check for (or download) the IPUMS data file and update config["ipums_data_path"] accordingly.
      6. Process the HCV eligibility data by calling process_hcv_eligibility with the updated configuration.
      7. After processing, delete the downloaded IPUMS file to prevent accumulation in the download folder.

    Parameters:
        config (dict): The global configuration dictionary containing:
            - "states": A list of state abbreviations.
            - "ipums_years": A list of years to process.
            - "data_dir": The base directory for data files.
            - "output_directory": The base output directory.
            - "hud_hcv_template": Template for the HUD HCV file path (which includes a {year} placeholder).
            - "api_settings": A dictionary of API settings (including "ipums_api_token" and "download_dir").
            - Other keys required by downstream processing functions.

    Returns:
        None
    """
    for state in config["states"]:
        for year in config["ipums_years"]:
            logging.info(f"Processing state: {state.upper()} for year: {year}")
            state_config = update_config_for_state(config, state)
            state_config["state"] = state.upper()
            state_config["year"] = year
            state_year_output = create_output_structure(config["output_directory"], state, year)
            state_config["output_directory"] = state_year_output
            state_config["hud_hcv_data_path"] = config["hud_hcv_template"].format(
                data_dir=config["data_dir"], state=state, year=year)
            ipums_file = get_ipums_data_file(state_config)
            state_config["ipums_data_path"] = ipums_file
            if ipums_file is None:
                logging.error(f"Skipping {state.upper()} {year} due to IPUMS data download failure.")
                continue
            state_config["ipums_data_path"] = ipums_file
            process_hcv_eligibility(state_config)
            try:
                if os.path.exists(state_config["ipums_data_path"]):
                    os.remove(state_config["ipums_data_path"])
                    logging.info(f"Deleted downloaded IPUMS file: {state_config['ipums_data_path']}")
            except Exception as e:
                logging.error(f"Error deleting downloaded IPUMS file: {e}")
