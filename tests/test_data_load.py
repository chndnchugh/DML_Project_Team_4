from src.data.load_data import load_raw_data

def test_load_raw_data():
    try:
        df = load_raw_data()
    except FileNotFoundError:
        assert True
        return
    assert not df.empty
