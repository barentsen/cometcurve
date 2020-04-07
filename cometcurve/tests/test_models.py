from ..models import comet_magnitude_power_law

def test_power_law():
    mag = comet_magnitude_power_law(h=10., n=1., delta=1., r=1.)
    assert mag == 10.