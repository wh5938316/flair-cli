from typing import Any, Dict, Optional, Union

from ._util import Arg, Opt, app, parse_config_overrides, show_validation_error


@app.command("info")
def info_cli(
    # fmt: off
        model: Optional[str] = Arg(None, help="Optional loadable spaCy pipeline"),
):
    info(model)
    # train(config_path, output_path, overrides=overrides)


def info(
    model: Optional[str] = None,
):
    print("info")
