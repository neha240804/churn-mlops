from pathlib import Path
import yaml


# Path to the configuration file
CONFIG_PATH = Path("config/config.yaml")


def load_config():
    """
    Load configuration from config/config.yaml.

    Returns:
        dict: Configuration dictionary.
    """

    # Check if config file exists
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found at: {CONFIG_PATH}"
        )

    # Read YAML file
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Check if YAML is empty
    if config is None:
        raise ValueError("Configuration file is empty.")

    return config


if __name__ == "__main__":
    config = load_config()

    print("Configuration Loaded Successfully!\n")
    print(config)