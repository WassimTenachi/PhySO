from . import physym
from . import learn
from . import task
from . import config
from . import benchmark

# Making important interface functions available at root level
fit = task.fit.fit
SR = task.sr.SR
ClassSR = task.class_sr.ClassSR
# User level flags
FLAG_USE_TEX = physym.program.FLAG_USE_TEX
# User level log loading tools
read_pareto_csv = benchmark.utils.read_logs.read_pareto_csv
read_pareto_pkl = learn.monitoring.read_pareto_pkl
