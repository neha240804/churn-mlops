from src.config import load_config


def test_load_config():

    config = load_config()

    assert "dataset" in config
    assert "artifacts" in config