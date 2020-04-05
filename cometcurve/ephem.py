from io import StringIO

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from astropy.time import Time
from astropy.utils.data import download_file


# Endpoint to obtain ephemerides from JPL/Horizons
HORIZONS_URL = ("https://ssd.jpl.nasa.gov/horizons_batch.cgi?"
                "batch=1&COMMAND=%27{target}%27&MAKE_EPHEM=%27YES%27%20&"
                "CENTER=%27500%27&TABLE_TYPE=%27OBSERVER%27&"
                "START_TIME=%27{start}%27&STOP_TIME=%27{stop}%27&"
                "STEP_SIZE=%27{step_size}%27%20&ANG_FORMAT=%27DEG%27&"
                "QUANTITIES=%2719,20,23%27&CSV_FORMAT=%27YES%27""")


class EphemFailure(Exception):
    # JPL/Horizons ephemerides could not be retrieved
    pass


def jpl2pandas(path):
    """Converts a csv ephemeris file from JPL/Horizons into a DataFrame.
    Parameters
    ----------
    path : str 
        Must be in JPL/Horizons' CSV-like format.
    Returns
    -------
    ephemeris : `pandas.DataFrame` object
    """
    jpl = open(path).readlines()
    csv_started = False
    csv = StringIO()
    for idx, line in enumerate(jpl):
        if line.startswith("$$EOE"):  # "End of ephemerides"
            break
        if csv_started:
            csv.write(line)
        if line.startswith("$$SOE"):  # "Start of ephemerides"
            csv.write(jpl[idx - 2])  # Header line
            csv_started = True
    if len(csv.getvalue()) < 1:
        jpl_output = "\n".join([line
                                for line in jpl])
        msg = jpl_output
        msg += ("Uhoh, something went wrong! "
                "Most likely, JPL/Horizons did not recognize the target."
                " Check their response above to understand why.")
        raise EphemFailure(msg)
    csv.seek(0)
    df = pd.read_csv(csv)
    # Simplify column names for user-friendlyness;
    # 'APmag' is the apparent magnitude which is returned for asteroids;
    # 'Tmag' is the total magnitude returned for comets:
    df.index.name = 'date'
    df = df.rename(columns={' Date__(UT)__HR:MN': "date",
                            '               r': 'r',
                            '             delta': 'delta',
                            '     S-O-T': 'elongation'})
    df['date'] = pd.to_datetime(df.date)
    df['jd'] = Time(df.date).jd
    return df


def get_ephemeris_file(target, start, stop, step_size=4, cache=True):
    """Returns a file-like object containing the JPL/Horizons response.
    Parameters
    ----------
    target : str
    start : str
    stop : str
    step_size : int
        Resolution of the ephemeris in number of days.
    Returns
    -------
    ephemeris : file-like object.
        Containing the response from JPL/Horizons.
    """
    arg = {
            "target": target.replace(" ", "%20"),
            "start": start,
            "stop": stop,
            "step_size": "{}%20d".format(step_size)
           }
    # If the target is a comet (i.e. name ends with "P"),
    # then we need to add the "CAP" directive to to select
    # the appropriate apparition.
    if target.endswith("P"):
        arg['target'] = "DES={}%3B%20CAP%3B".format(arg['target'])
    if step_size < 1:  # Hack: support step-size in hours
        arg['step_size'] = "{:.0f}%20h".format(step_size * 24)
    print("Obtaining ephemeris for {target} "
          "from JPL/Horizons...".format(**arg))
    url = HORIZONS_URL.format(**arg)
    return download_file(url, cache=cache)


def get_ephemeris(target, first, last, step_size=2, cache=True):
    """Returns the ephemeris dataframe for a single campaign."""
    path = get_ephemeris_file(target, first, last, step_size, cache=cache)
    return jpl2pandas(path)


def create_sun_distance_func(ephemeris):
    sun_distance_func = interp1d(ephemeris.date.values.astype(float),
                                 ephemeris.r,
                                 kind='quadratic',
                                 fill_value="extrapolate")
    return sun_distance_func


def create_earth_distance_func(ephemeris):
    earth_distance_func = interp1d(ephemeris.date.values.astype(float),
                                   ephemeris.delta,
                                   kind='quadratic',
                                   fill_value="extrapolate")
    return earth_distance_func