import inspect
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

from .config import Config
from .errors import Errors

# Default order of sections in the config file. Not all sections needs to exist,
# and additional sections are added at the end, in alphabetical order.
CONFIG_SECTION_ORDER = [
    "paths",
    "variables",
    "system",
    "nlp",
    "components",
    "corpora",
    "training",
    "pretraining",
    "initialize",
]


class SimpleFrozenList(list):
    """Wrapper class around a list that lets us raise custom errors if certain
    attributes/methods are accessed. Mostly used for properties like
    Language.pipeline that return an immutable list (and that we don't want to
    convert to a tuple to not break too much backwards compatibility). If a user
    accidentally calls nlp.pipeline.append(), we can raise a more helpful error.
    """

    def __init__(self, *args, error: str = Errors.E104) -> None:
        """Initialize the frozen list.

        error (str): The error message when user tries to mutate the list.
        """
        self.error = error
        super().__init__(*args)

    def append(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def clear(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def extend(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def insert(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def pop(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def remove(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def reverse(self, *args, **kwargs):
        raise NotImplementedError(self.error)

    def sort(self, *args, **kwargs):
        raise NotImplementedError(self.error)


class SimpleFrozenDict(dict):
    """Simplified implementation of a frozen dict, mainly used as default
    function or method argument (for arguments that should default to empty
    dictionary). Will raise an error if user or spaCy attempts to add to dict.
    """

    def __init__(self, *args, error: str = Errors.E002, **kwargs) -> None:
        """Initialize the frozen dict. Can be initialized with pre-defined
        values.

        error (str): The error message when user tries to assign to dict.
        """
        super().__init__(*args, **kwargs)
        self.error = error

    def __setitem__(self, key, value):
        raise NotImplementedError(self.error)

    def pop(self, key, default=None):
        raise NotImplementedError(self.error)

    def update(self, *other):
        raise NotImplementedError(self.error)


def ensure_path(path: Any) -> Any:
    """Ensure string is converted to a Path.

    path (Any): Anything. If string, it's converted to Path.
    RETURNS: Path or original argument.
    """
    if isinstance(path, str):
        return Path(path)
    else:
        return path


def load_config(
    path: Union[str, Path],
    overrides: Dict[str, Any] = None,
    interpolate: bool = False,
) -> Config:
    """Load a config file. Takes care of path validation and section order.

    path (Union[str, Path]): Path to the config file or "-" to read from stdin.
    overrides: (Dict[str, Any]): Config overrides as nested dict or
        dict keyed by section values in dot notation.
    interpolate (bool): Whether to interpolate and resolve variables.
    RETURNS (Config): The loaded config.
    """
    config_path = ensure_path(path)
    config = Config(section_order=CONFIG_SECTION_ORDER)
    overrides = overrides or SimpleFrozenDict()
    if str(config_path) == "-":  # read from standard input
        return config.from_str(
            sys.stdin.read(), overrides=overrides, interpolate=interpolate
        )
    else:
        if not config_path or not config_path.is_file():
            raise IOError(Errors.E001.format(path=config_path, name="config file"))
        return config.from_disk(
            config_path, overrides=overrides, interpolate=interpolate
        )


def copy_config(config: Union[Dict[str, Any], Config]) -> Config:
    """Deep copy a Config. Will raise an error if the config contents are not
    JSON-serializable.

    config (Config): The config to copy.
    RETURNS (Config): The copied config.
    """
    try:
        return Config(config).copy()
    except ValueError:
        raise ValueError(Errors.E103.format(config=config)) from None


def raise_error(proc_name, proc, docs, e):
    raise e


def get_object_name(obj: Any) -> str:
    """Get a human-readable name of a Python object, e.g. a pipeline component.

    obj (Any): The Python object, typically a function or class.
    RETURNS (str): A human-readable name.
    """
    if hasattr(obj, "name") and obj.name is not None:
        return obj.name
    if hasattr(obj, "__name__"):
        return obj.__name__
    if hasattr(obj, "__class__") and hasattr(obj.__class__, "__name__"):
        return obj.__class__.__name__
    return repr(obj)


def dot_to_dict(values: Dict[str, Any]) -> Dict[str, dict]:
    """Convert dot notation to a dict. For example: {"token.pos": True,
    "token._.xyz": True} becomes {"token": {"pos": True, "_": {"xyz": True }}}.

    values (Dict[str, Any]): The key/value pairs to convert.
    RETURNS (Dict[str, dict]): The converted values.
    """
    result: Dict[str, dict] = {}
    for key, value in values.items():
        path = result
        parts = key.lower().split(".")
        for i, item in enumerate(parts):
            is_last = i == len(parts) - 1
            path = path.setdefault(item, value if is_last else {})
    return result


def is_same_func(func1: Callable, func2: Callable) -> bool:
    """Approximately decide whether two functions are the same, even if their
    identity is different (e.g. after they have been live reloaded). Mostly
    used in the @Language.component and @Language.factory decorators to decide
    whether to raise if a factory already exists. Allows decorator to run
    multiple times with the same function.

    func1 (Callable): The first function.
    func2 (Callable): The second function.
    RETURNS (bool): Whether it's the same function (most likely).
    """
    if not callable(func1) or not callable(func2):
        return False
    if not hasattr(func1, "__qualname__") or not hasattr(func2, "__qualname__"):
        return False
    same_name = func1.__qualname__ == func2.__qualname__
    same_file = inspect.getfile(func1) == inspect.getfile(func2)
    same_code = inspect.getsourcelines(func1) == inspect.getsourcelines(func2)
    return same_name and same_file and same_code


def get_arg_names(func: Callable) -> List[str]:
    """Get a list of all named arguments of a function (regular,
    keyword-only).

    func (Callable): The function
    RETURNS (List[str]): The argument names.
    """
    argspec = inspect.getfullargspec(func)
    return list(dict.fromkeys([*argspec.args, *argspec.kwonlyargs]))


class Debug:
    def __init__(self, logger, default_value: bool = False):
        self.logger = logger
        self._debug = default_value

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, new_value):
        self.set_debug(new_value)

    def set_debug(self, value):
        if not isinstance(value, bool):
            raise TypeError("Debug can only be set to bool.")
        self._debug = value
        if self.debug:
            self.logger.warning("Debug mode turned ON!")

    def __eq__(self, other):
        if not isinstance(other, bool):
            raise TypeError("Debug can only be compared to bool.")
        return self.debug == other

    def __bool__(self):
        return self._debug is True

    def __str__(self):
        return "on" if self.debug else "off"

    def __repr__(self):
        return f"NLPy debug mode is: {'on' if self.debug else 'off'}"
