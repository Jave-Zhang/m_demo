import json
import os
import munch
import random
import numpy as np

def get_config_from_json(json_file):
    """
    Get the configuration from a JSON file
    
    Args:
        json_file (str): Path to the configuration JSON file
        
    Returns:
        dict: Configuration dictionary loaded from the JSON file
    """
    # Parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    return config_dict


def process_config(jsonfile):
    """
    Process the configuration file and create a config object
    
    Args:
        jsonfile (str): Path to the configuration JSON file
        
    Returns:
        munch.Munch: Configuration object with attribute-style access
    """
    config_dict = get_config_from_json(jsonfile)
    config = munch.Munch(config_dict)
    
    return config


