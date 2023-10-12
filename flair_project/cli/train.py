from pathlib import Path
from typing import Any, Dict, Optional, Union

import flair
import typer
from flair.trainers import ModelTrainer
from wasabi import msg

import flair_project.utils as utils
from flair_project.config import registry

from ._util import Arg, Opt, app, parse_config_overrides, show_validation_error


@app.command(
    "train", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def train_cli(
    # fmt: off
        ctx: typer.Context,  # This is only used to read additional arguments
        config_path: Path = Arg(..., help="Path to config file", exists=True, allow_dash=True),
        output_path: Optional[Path] = Opt(None, "--output", "--output-path", "-o",
                                          help="Output directory to store trained pipeline in"),
):
    overrides = parse_config_overrides(ctx.args)
    train(config_path, output_path, overrides=overrides)


def train(
    config_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    *,
    overrides: Optional[Dict[str, Any]] = None,
):
    config_path = utils.ensure_path(config_path)
    output_path = utils.ensure_path(output_path)
    overrides = overrides or utils.SimpleFrozenDict()
    # Make sure all files and paths exists if they are needed
    if not config_path or (str(config_path) != "-" and not config_path.exists()):
        msg.fail("Config file not found", config_path, exits=1)
    if not output_path:
        msg.info("No output directory provided")
    else:
        if not output_path.exists():
            output_path.mkdir(parents=True)
            msg.good(f"Created output directory: {output_path}")
        msg.info(f"Saving to output directory: {output_path}")

    with show_validation_error(config_path):
        config = utils.load_config(config_path, overrides=overrides, interpolate=False)

    resolved = registry.resolve(config)
    training_config = resolved["training"]

    if "seed" in resolved["system"]:
        flair.set_seed(resolved["system"]["seed"])

    trainer = ModelTrainer(training_config["model"], training_config["corpus"])
    lr = resolved["training"].get("lr", 0.1)

    trainer.fine_tune(
        output_path,
        learning_rate=lr,
        mini_batch_size=training_config.get("batch_size", 1),
        max_epochs=training_config.get("max_epochs", 1),
        optimizer=training_config.get("optimizer"),
        scheduler=training_config.get("scheduler"),
    )
