import os


def test_model_exists():

    assert os.path.exists("data/artifacts/model.pkl")