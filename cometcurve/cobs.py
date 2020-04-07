"""Module for interacting with the Comet Observations Database (COBS)."""
import numpy as np
import pandas as pd
from astropy.time import Time

from . import PACKAGEDIR, log


# Column numbers of COBS/ICQ data fields
COLUMNS = {
        'comet': (0, 11),
        'date': (11, 21),
        'fractional_day': (21, 24),
        'method': (26, 27),
        'upper_limit': (27, 28),
        'magnitude': (28, 32),
        'poor': (32, 33),
        'aperture': (35, 40),
        'instrument': (40, 41),
        'observer': (75, 80),
        'comments': (130, -1)
    }


class CometObservations():
    """Class to interact with the Comet Observation Database (COBS).
    
    Parameters
    ----------
    data : `pandas.DataFrame`
        DataFrame returned by `read_cobs()`.
    """
    def __init__(self, data=None):
        if data is None:
            self.data = read_cobs()
        else:
            self.data = data

    def get_observer_list(self):
        """Returns a string listing all observer names.
        
        The names are sorted by number of observations.
        """
        return ", ".join(self.data.observer_name.value_counts().keys())


def read_cobs(filename=None, comet=None, start=None, stop=None,
              allowed_methods=('S', 'B', 'M', 'I', 'E', 'Z', 'V', 'O'),):
    """Returns a `CometObservations` instance containing the COBS database."""
    # Default filename
    if filename is None:
        filename = [PACKAGEDIR / 'data/cobs-data-before-2020.txt',
                    PACKAGEDIR / 'data/cobs-data-2020.txt']

    # Read the data
    data = []
    for fn in tuple(filename):
        log.info(f"Reading {fn}")
        data.append(pd.read_fwf(fn,
                                colspecs=list(COLUMNS.values()),
                                header=None,
                                names=COLUMNS.keys()))
    df = pd.concat(data)

    # Remove bad lines defined as follows:
    # * date i.e. year does not start with the character 1 or 2  (indicative of ill-formatted ICQ)
    # * the magnitude is not missing (character "-")
    # * the "poor" column is empty (note: this removes a small number of entries
    #   for comet 1965S1 where magnitude -10.0 overflows into the poor column).
    # * the magnitude is not an upper limit (`df.upper_limit.isna()`)
    # * did not use a convential method (cf. `allowed_methods`), i.e. a method
    #   that does not yield something similar to a V-band integrated magnitude.
    bad_data_mask = (df.date.str[0].isin(["1","2"])
                     & (df.magnitude != '-')
                     & df.poor.isna()
                     & df.upper_limit.isna())
    if allowed_methods != 'all':
        bad_data_mask &= df.method.isin(allowed_methods)
    df = df[bad_data_mask]

    df['time'] = pd.to_datetime(df.date, utc=True) + pd.to_timedelta(df.fractional_day, unit='D')
    df['jd'] = Time(df.time).jd
    df['magnitude'] = df.magnitude.astype(float)
    df['aperture'] = df.aperture.astype(float)
    df['visual'] = df.method.isin(('S', 'M', 'B'))
    df['binocular'] = df.instrument == 'B'
    df['poor'] = df.poor == ":"
    df['observer_name'] = df.comments.str.split(pat="[,;]", expand=True)[0]

    # Optional data filtering
    mask = np.ones(len(df), dtype=bool)
    if comet is not None:
        mask &= df.comet == comet.replace(" ", "")
    if start is not None:
        mask &= df.time > start
    if stop is not None:
        mask &= df.time < stop        
    df = df[mask]

    # Add a column detailing the number of observations by each observer
    df_counts = df.observer.value_counts().reset_index()
    df_counts.columns = ['observer', 'observations']
    df = pd.merge(df, df_counts, on="observer")

    return CometObservations(df)
