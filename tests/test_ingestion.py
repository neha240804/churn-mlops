from src.data_ingestion import get_data


def test_dataset_loaded():

    df = get_data()

    assert df.shape[0] > 0
    assert df.shape[1] == 21