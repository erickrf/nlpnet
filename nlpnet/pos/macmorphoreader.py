from __future__ import unicode_literals

import warnings

# backwards compatibility
from .pos_reader import *

warnings.warn('Module macmorphoreader is deprecated. Use module pos_reader instead.')
