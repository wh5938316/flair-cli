import inspect
from typing import Callable, Dict, List, Optional, Union

import catalogue
import confection
from catalogue import Registry, RegistryError
from confection import VARIABLE_RE, Config, ConfigValidationError, Promise

from .errors import Errors
from .types import Decorator


class ENV_VARS:
    CONFIG_OVERRIDES = "FLAIR_CONFIG_OVERRIDES"


class registry(confection.registry):
    factories = catalogue.create("flair", "internal_factories")
    misc = catalogue.create("flair", "misc", entry_points=True)
    architectures = catalogue.create("flair", "architectures", entry_points=True)
    reader = catalogue.create("flair", "reader", entry_points=True)
    models: Decorator = catalogue.create("flair", "models", entry_points=True)
    cli = catalogue.create("flair", "cli", entry_points=True)
    optimizers: Decorator = catalogue.create("flair", "optimizers", entry_points=True)
    schedulers: Decorator = catalogue.create("flair", "schedulers", entry_points=True)

    @classmethod
    def create(cls, registry_name: str, entry_points: bool = False) -> None:
        """Create a new custom registry."""
        if hasattr(cls, registry_name):
            raise ValueError(f"Registry '{registry_name}' already exists")
        reg: Decorator = catalogue.create(
            "thinc", registry_name, entry_points=entry_points
        )
        setattr(cls, registry_name, reg)

    @classmethod
    def get_registry_names(cls) -> List[str]:
        """List all available registries."""
        names = []
        for name, value in inspect.getmembers(cls):
            if not name.startswith("_") and isinstance(value, Registry):
                names.append(name)
        return sorted(names)

    @classmethod
    def get(cls, registry_name: str, func_name: str) -> Callable:
        """Get a registered function from the registry."""
        if not hasattr(cls, registry_name):
            names = ", ".join(cls.get_registry_names()) or "none"
            raise RegistryError(Errors.E101.format(name=registry_name, available=names))
        reg = getattr(cls, registry_name)
        try:
            func = reg.get(func_name)
        except RegistryError:
            if func_name.startswith("nlp."):
                legacy_name = func_name.replace("nlp.", "nlp-legacy.")
                try:
                    return reg.get(legacy_name)
                except catalogue.RegistryError:
                    pass
            available = ", ".join(sorted(reg.get_all().keys())) or "none"
            raise RegistryError(
                Errors.E102.format(
                    name=func_name, reg_name=registry_name, available=available
                )
            ) from None
        return func

    @classmethod
    def find(
        cls, registry_name: str, func_name: str
    ) -> Dict[str, Optional[Union[str, int]]]:
        """Find information about a registered function, including the
        module and path to the file it's defined in, the line number and the
        docstring, if available.

        registry_name (str): Name of the catalogue registry.
        func_name (str): Name of the registered function.
        RETURNS (Dict[str, Optional[Union[str, int]]]): The function info.
        """
        # We're overwriting this classmethod so we're able to provide more
        # specific error messages and implement a fallback to spacy-legacy.
        if not hasattr(cls, registry_name):
            names = ", ".join(cls.get_registry_names()) or "none"
            raise RegistryError(Errors.E101.format(name=registry_name, available=names))
        reg = getattr(cls, registry_name)
        try:
            func_info = reg.find(func_name)
        except RegistryError:
            if func_name.startswith("nlp."):
                legacy_name = func_name.replace("nlp.", "nlp-legacy.")
                try:
                    return reg.find(legacy_name)
                except catalogue.RegistryError:
                    pass
            available = ", ".join(sorted(reg.get_all().keys())) or "none"
            raise RegistryError(
                Errors.E102.format(
                    name=func_name, reg_name=registry_name, available=available
                )
            ) from None
        return func_info

    @classmethod
    def has(cls, registry_name: str, func_name: str) -> bool:
        """Check whether a function is available in a registry."""
        if not hasattr(cls, registry_name):
            return False
        reg = getattr(cls, registry_name)
        if func_name.startswith("nlp."):
            legacy_name = func_name.replace("nlp.", "nlp-legacy.")
            return func_name in reg or legacy_name in reg
        return func_name in reg


__all__ = [
    "Config",
    "registry",
    "ConfigValidationError",
    "Promise",
    "VARIABLE_RE",
    "ENV_VARS",
]
