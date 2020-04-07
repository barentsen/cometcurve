"""Module for interacting with the Comet Observations Database (COBS)."""
from io import StringIO
import re
from pathlib import Path

from appdirs import user_cache_dir
from astropy.time import Time
import mechanize
import numpy as np
import pandas as pd

from . import PACKAGEDIR, log


# Where to store COBS data?
CACHEDIR = Path(user_cache_dir("cometcurve"))

# Column numbers of COBS/ICQ data fields
ICQ_COLUMNS = {
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


def read_cobs(years=('2020'), comet=None, start=None, stop=None,
              allowed_methods=('S', 'B', 'M', 'I', 'E', 'Z', 'V', 'O'),):
    """Returns a `CometObservations` instance containing the COBS database."""
    if years == 'all':
        years = tuple(range(2018, 2020))

    # Read the data
    data = []
    for yr in np.atleast_1d(years):
        try:
            dfyr = _get_cache_dataframe(yr)
        except FileNotFoundError:
            dfyr = download_cobs(yr)
        data.append(dfyr)
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
                     & df.poor.isna()
                     & df.upper_limit.isna())
    if df.magnitude.dtype is not float:
        bad_data_mask &= df.magnitude != '-'
    if allowed_methods != 'all':
        bad_data_mask &= df.method.isin(allowed_methods)
    df = df[bad_data_mask]

    df['time'] = pd.to_datetime(df.date, utc=True) + pd.to_timedelta(df.fractional_day, unit='D')
    df['jd'] = Time(df.time).jd
    df['magnitude'] = df.magnitude.astype(float)
    df['aperture'] = df.aperture.astype(float)
    df['visual'] = df.method.isin(('S', 'M', 'B'))
    df['binocular'] = df.instrument == 'B'
    df['poor'] = df.poor.astype(str) == ":"
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


def _parse_icq(fileobj):
    """Parse a International Comet Quarterly (ICQ) format file."""
    df = pd.read_fwf(fileobj, colspecs=list(ICQ_COLUMNS.values()),
                     names=ICQ_COLUMNS.keys(), header=None)
    return df


def _get_cache_filename(year=2020):
    """Returns the `Path` to the COBS data file for a given year."""
    return CACHEDIR / f'cobs{year}.feather'


def _get_cache_dataframe(year=2020):
    fn = _get_cache_filename(year)
    if fn.exists():
        log.info(f"Loading {fn}")
        return pd.read_feather(fn)
    else:
        raise FileNotFoundError(f"File not found: {fn}")


def download_cobs(year=2020, update=False):
    """Download a year of COBS data and save it in the cache."""
    URL = "https://cobs.si/analysis"
    cache_fn = _get_cache_filename(year)
    if cache_fn.exists() and not update:
        raise IOError(f"Data for {year} has already been downloaded. "
                      "Use `update=True` to download again.")
    log.info(f"Retrieving {year} data from {URL}")
    br = mechanize.Browser()
    br.set_handle_robots(False)
    br.open(URL)
    br.select_form(nr=0)
    br.form['START_DATE'] = f'{year}/01/01 00:00'
    br.form['END_DATE'] = f'{year}/12/31 00:00'
    br.submit(id="getobs")
    resp = None
    for link in br.links():
        match = re.compile('.*lightcurve_.*.dat').search(link.url)
        if match:
            log.info(f"Downloading {link.url}")
            resp = br.follow_link(link)
            break
    if resp is None:
        raise IOError(f"Could not download COBS data for {year}.")
    # Parse the format and save to a feather cache file
    df = _parse_icq(StringIO(resp.get_data().decode()))
    cache_fn.parent.mkdir(exist_ok=True)
    log.info(f"Saving data to {cache_fn}")
    df.to_feather(cache_fn)
    return df
