from . import physym
from . import learn
from . import task
from . import config
from . import benchmark

# Making important interface functions available at root level
fit = task.fit.fit
SR = task.sr.SR
ClassSR = task.class_sr.ClassSR

# User level log loading tools
read_pareto_csv = benchmark.utils.read_logs.read_pareto_csv
read_pareto_pkl = learn.monitoring.read_pareto_pkl

# Exposed version
from ._version import __version__