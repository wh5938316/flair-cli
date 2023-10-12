from flair_project.config import registry
from flair.optim import LinearSchedulerWithWarmup, SGDW, ExpAnnealLR, ReduceLRWDOnPlateau


@registry.schedulers("flair.SGDW.v1")
def create_sgdw():
    return SGDW


@registry.schedulers("flair.LinearSchedulerWithWarmup.v1")
def create_linear_scheduler_with_warmup():
    return LinearSchedulerWithWarmup


@registry.schedulers("flair.ExpAnnealLR.v1")
def create_exp_anneal_lr():
    return ExpAnnealLR


@registry.schedulers("flair.ReduceLRWDOnPlateau.v1")
def create_reduce_lrwd_on_plateau():
    return ReduceLRWDOnPlateau
