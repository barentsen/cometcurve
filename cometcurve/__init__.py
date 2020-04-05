from pathlib import Path
PACKAGEDIR = Path(__file__).parent
MPLSTYLE = PACKAGEDIR / 'data/cometcurve.mplstyle'

import logging
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())

from .version import __version__
from .cobs import *
from .ephem import *
from .models import *
