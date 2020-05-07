import pymc3 as pm
import matplotlib.pyplot as pl
from matplotlib import dates
import numpy as np
import pandas as pd

from . import ephem, MPLSTYLE, PACKAGEDIR
from .cobs import read_cobs


class CometModel():

    def __init__(self, comet="2019 Y4", cobs_id=None, horizons_id=None,
                 start="2020-01-01", stop="2020-08-01"):
        self.cobs_id = comet if cobs_id is None else cobs_id
        self.horizons_id = comet if horizons_id is None else horizons_id
        self.comet = comet
        self.start = start
        self.stop = stop
        # Load the observations
        self.obs = read_cobs(comet=self.cobs_id, start=self.start, stop=self.stop)
        # Load the ephemeris
        self.ephem = ephem.get_ephemeris(self.horizons_id, self.start, self.stop)
        self.sun_distance_func = ephem.create_sun_distance_func(self.ephem)
        self.earth_distance_func = ephem.create_earth_distance_func(self.ephem)
        # Initialize
        self.model = None
        self.trace = None

    def sample(self, draws=500, cores=4):
        if self.model is None:
            self.model = self.create_pymc_model()
        with self.model:
            trace = pm.sample(draws=draws, cores=cores)
        self.trace = trace
        return trace

    def traceplot(self, var_names=('n', 'h', 'beta')):
        return pm.traceplot(self.trace, var_names=var_names);

    def get_parameter_summary(self, params=('n', 'h', 'beta')):
        result = {'comet': self.comet}
        for par in params:
            partrace = self.trace.get_values(par)
            result[f'{par}_mean'] = np.nanmean(partrace)
            result[f'{par}_std'] = np.nanstd(partrace)
            for percentile in [0.1, 16, 50, 84, 0.5, 99.9]:
                result[f'{par}_{percentile:.1f}p'] = np.nanpercentile(partrace, percentile)
        return result

    def mean_model(self, times):
        trace_n = self.trace.get_values('n')
        trace_h = self.trace.get_values('h')
        mean_h = np.nanmean(trace_h)
        mean_n = np.nanmean(trace_n)
        return comet_magnitude_power_law(h=mean_h,
                                         delta=self.earth_distance_func(times),
                                         n=mean_n,
                                         r=self.sun_distance_func(times))

    def plot(self, title=None, show_year=False, min_elongation=None):
        """Plot the model alongside the observations."""
        if title is None:
            title = f"Light curve of Comet {self.comet}"

        if self.trace is None:
            self.trace = self.sample()

        with pl.style.context(str(MPLSTYLE)):
            ax = pl.subplot(111)
          
            times = np.arange(self.ephem.date.values[0],
                              self.ephem.date.values[-1] + 1_000_000_000_000_000,
                              np.timedelta64(12, 'h'))

            # Show the observations
            pl.scatter(self.obs.data[self.obs.data.visual].time,
                       self.obs.data[self.obs.data.visual].magnitude,
                       marker='+', lw=0.7, s=40, label="Visual observations (COBS)",
                       c='#2980b9', alpha=0.8, zorder=50)
            pl.scatter(self.obs.data[~self.obs.data.visual].time,
                       self.obs.data[~self.obs.data.visual].magnitude,
                       marker='x', lw=0.7, s=30, label="CCD observations (COBS)",
                       c='#c0392b', alpha=0.8, zorder=50)

            # Show the model fit
            trace_n = self.trace.get_values('n')
            trace_h = self.trace.get_values('h')
            mean_h = np.nanmean(trace_h)
            mean_n = np.nanmean(trace_n)

            # Show the uncertainty based on 100 samples
            if len(trace_n) < 100:
                step = 1
            else:
                step = int(len(trace_n)/100)
            for idx in range(0, len(trace_n), step):
                model = comet_magnitude_power_law(h=trace_h[idx],
                                                  delta=self.earth_distance_func(times),
                                                  n=trace_n[idx],
                                                  r=self.sun_distance_func(times))
                pl.plot(times, model, c='black', lw=1, alpha=0.02, zorder=10)    

            model = comet_magnitude_power_law(h=mean_h,
                                              delta=self.earth_distance_func(times),
                                              n=mean_n,
                                              r=self.sun_distance_func(times))
            pl.plot(times, model, ls='dashed', c='black', lw=2,
                    label=f"Model (H={mean_h:.1f}; n={mean_n:.1f})", zorder=20)

            # Show the elongation black-out
            if min_elongation is not None:
                d1 = self.ephem[self.ephem.elongation < min_elongation].date.min()
                d2 = self.ephem[self.ephem.elongation < min_elongation].date.max()
                ax.fill_between([d1, d2], -30, 30,
                                label=f'Elongation < {min_elongation}°',
                                hatch='//////', facecolor='None',
                                edgecolor='#bdc3c7', zorder=5)

            if show_year:
                ax.xaxis.set_major_formatter(dates.DateFormatter('%-d %b %Y'))
                pl.xlabel("Date")                
            else:
                ax.xaxis.set_major_formatter(dates.DateFormatter('%-d %b'))
                pl.xlabel(f"Date ({self.start[:4]})")

            ax.xaxis.set_minor_locator(dates.AutoDateLocator(minticks=50, maxticks=100))
            labels = ax.get_xmajorticklabels()
            pl.setp(labels, rotation=45) 
            labels = ax.get_xminorticklabels()
            pl.setp(labels, rotation=45)

            pl.ylim([int(np.max(model)+3), int(np.min(model-6))])
            pl.xlim([np.datetime64(self.start), np.datetime64(self.stop)])
            pl.ylabel("Magnitude")
            pl.title(title)
            handles, labels = ax.get_legend_handles_labels()
            try:
                pl.legend([handles[x] for x in (1, 2, 0, 3)],
                        [labels[x] for x in (1, 2, 0, 3)])
            except IndexError:
                pl.legend()
            pl.tight_layout()
        return ax

    def create_pymc_model(self, min_observations=0, observer_bias=False):
        """Returns a PyMC3 model."""
        dfobs = self.obs.data[self.obs.data.observations > min_observations]
        
        with pm.Model() as model:
            delta = self.earth_distance_func(dfobs.time.values)
            r = self.sun_distance_func(dfobs.time.values)
            n = pm.Normal('n', mu=3.49, sigma=1.36)  # activity parameter
            h = pm.Normal('h', mu=6.66, sigma=1.98)  # absolute magnitude
            
            model_mag = comet_magnitude_power_law(h=h, n=n, delta=delta, r=r)

            if observer_bias == True:
                observers = dfobs.observer.unique()
                for obs in observers:
                    mask = np.array(dfobs.observer.values == obs) 
                    beta = pm.HalfNormal('beta_'+obs, sigma=0.5)
                    bias = pm.Normal('bias_'+obs, mu=0., sigma=.5)
                    obsmag = pm.Cauchy('obsmag_'+obs,
                                       alpha=model_mag[mask] + bias,
                                       beta=beta,
                                       observed=dfobs.magnitude[mask])
            else:
                beta = 0.47 + pm.HalfNormal('beta', sigma=0.02)
                obsmag = pm.Cauchy('obsmag', alpha=model_mag,
                                   beta=beta, observed=dfobs.magnitude)

        self.model = model
        return self.model


def comet_magnitude_power_law(h=10., n=4., delta=1., r=1.):
    """The conventional power-law formula to predict a comet's brightness.

    Parameters
    ----------
    h : float
        Absolute magnitude.
    n : float
        Activity parameter.
    delta : float
        Comet's geocentric distance in AU.
    r : float
        Comet's heliocentric distance in AU.
    """
    return h + 5*np.log10(delta) + 2.5*n*np.log10(r)


def fit_all_comets():
    comets = pd.read_csv(PACKAGEDIR / "data/comets.csv")
    result = []
    for comet in comets.itertuples():
        print(comet.comet)
        model = CometModel(comet=comet.comet,
                          cobs_id=comet.cobs_id,
                          horizons_id=comet.horizons_id,
                          start=comet.start,
                          stop=comet.stop)
        model.sample(draws=300)
        ax = model.plot()
        output_fn = f"output/{comet.cobs_id}.png"
        print(f"Writing {output_fn}")
        ax.figure.savefig(output_fn)
        pl.close()
        params = model.get_parameter_summary()
        result.append(params)
    df = pd.DataFrame(result)
    print(f"Prior for n: mean={df.n_mean.mean():.2f}, std={df.n_mean.std():.2f}")
    print(f"Prior for h: mean={df.h_mean.mean():.2f}, std={df.h_mean.std():.2f}")
    print(f"Prior for beta: mean={df.beta_mean.mean():.2f}, std={df.beta_mean.std():.2f}")
    return df


def fit_all_comets2():
    result = []
    df = read_cobs()
    comet_counts = df[df.date > '1990'].comet.value_counts()
    well_observed_comets = comet_counts[comet_counts > 300].index
    well_observed_mask = df.comet.isin(well_observed_comets)
    for comet in well_observed_comets:
        print(comet)
        model = CometModel(comet=comet.comet,
                          cobs_id=comet.cobs_id,
                          horizons_id=comet.horizons_id,
                          start=comet.start,
                          stop=comet.stop)
        model.sample(draws=300)
        ax = model.plot()
        output_fn = f"output/{comet.cobs_id}.png"
        print(f"Writing {output_fn}")
        ax.figure.savefig(output_fn)
        pl.close()
        params = model.get_parameter_summary()
        result.append(params)
    df = pd.DataFrame(result)
    print(f"Prior for n: mean={df.n_mean.mean():.2f}, std={df.n_mean.std():.2f}")
    print(f"Prior for h: mean={df.h_mean.mean():.2f}, std={df.h_mean.std():.2f}")
    print(f"Prior for beta: mean={df.beta_mean.mean():.2f}, std={df.beta_mean.std():.2f}")
    return df
