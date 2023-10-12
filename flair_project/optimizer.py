from flair import logger
from flair.training_utils import log_line
from torch.optim import AdamW
from flair_project.config import registry

patterns_optimizer = {
    "additional_layers": ["additional"],
    "top_layer": ["additional", "bert_model.encoder.layer.11."],
    "top4_layers": [
        "additional",
        "bert_model.encoder.layer.11.",
        "encoder.layer.10.",
        "encoder.layer.9.",
        "encoder.layer.8",
    ],
    "all_encoder_layers": ["additional", "bert_model.encoder.layer"],
    "all": ["additional", "bert_model.encoder.layer", "bert_model.embeddings"],
}


def get_bert_params(models, type_optimization: str):
    """Optimizes the network with AdamWithDecay"""
    if type_optimization not in patterns_optimizer:
        print(
            "Error. Type optimizer must be one of %s" % (str(patterns_optimizer.keys()))
        )
    parameters_with_decay = []
    parameters_with_decay_names = []
    parameters_without_decay = []
    parameters_without_decay_names = []
    no_decay = ["bias", "gamma", "beta"]
    patterns = patterns_optimizer[type_optimization]

    for model in models:
        for n, p in model.named_parameters():
            if any(t in n for t in patterns):
                if any(t in n for t in no_decay):
                    parameters_without_decay.append(p)
                    parameters_without_decay_names.append(n)
                else:
                    parameters_with_decay.append(p)
                    parameters_with_decay_names.append(n)

    log_line(logger)
    logger.info("The following parameters will be optimized WITH decay:")
    logger.info(ellipses(parameters_with_decay_names, 5, " , "))
    log_line(logger)
    logger.info("The following parameters will be optimized WITHOUT decay:")
    logger.info(ellipses(parameters_without_decay_names, 5, " , "))
    log_line(logger)

    optimizer_grouped_parameters = [
        {"params": parameters_with_decay, "weight_decay": 0.01},
        {"params": parameters_without_decay, "weight_decay": 0.0},
    ]

    return optimizer_grouped_parameters


def ellipses(lst, max_display=5, sep="|"):
    """
    Like join, but possibly inserts an ellipsis.
    :param lst: The list to join on
    :param int max_display: the number of items to display for ellipsing.
        If -1, shows all items
    :param string sep: the delimiter to join on
    """
    # copy the list (or force it to a list if it's a set)
    choices = list(lst)
    # insert the ellipsis if necessary
    if 0 < max_display < len(choices):
        ellipsis_tail = "...and {} more".format(len(choices) - max_display)
        choices = choices[:max_display] + [ellipsis_tail]
    return sep.join(str(c) for c in choices)


@registry.optimizers("flair.AdamW.v1")
def create_adamw(
        params,
        lr: float,
):
    return AdamW(params, lr=lr)


@registry.misc("flair.GetBertParams.v1")
def create_get_bert_optimizer(
    model,
    type_optimization: str = "all_encoder_layers",
):
    return get_bert_params([model], type_optimization)
