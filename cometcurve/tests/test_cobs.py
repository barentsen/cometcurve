from ..cobs import read_cobs


def test_read():
    df = read_cobs()
    # COBS has more than 240,000 entries!
    assert len(df) > 240000


def test_read_with_filter():
    df = read_cobs(start="2020-01-01")
    assert len(df[df.time < "2020-01-01"]) == 0
