from src.config import load_config

def test_dataset_path_exists():

    config = load_config()

    assert "raw_data_path" in config["dataset"]