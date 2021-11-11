import json

def load_settings(path_settings):
    """Loads the settings json as dictionary

    Args:
        path_settings (string): the path to the settings

    Returns:
        dict: the settings as dictionary
    """
    try:
        with open(path_settings, 'r') as settings_file:
            settings = json.load(settings_file)
    except(FileNotFoundError):
        print('Error: No "settings.json" found.')

    return settings