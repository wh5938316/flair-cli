class ErrorsWithCodes(type):
    def __getattribute__(self, code):
        msg = super().__getattribute__(code)
        if code.startswith("__"):  # python system attributes like __class__
            return msg
        else:
            return "[{code}] {msg}".format(code=code, msg=msg)


class Errors(metaclass=ErrorsWithCodes):
    # system level error
    E001 = "Could not read {name} from {path}"
    E002 = (
        "Can't write to frozen dictionary. This is likely an internal "
        "error. Are you writing to a default function argument?"
    )
    # config & cli error
    E101 = "Unknown function registry: '{name}'.\n\nAvailable names: {available}"
    E102 = (
        "Could not find function '{name}' in function registry '{reg_name}'. "
        "If you're using a custom function, make sure the code is available. "
        "If the function is provided by a third-party package, e.g. , "
        "make sure the package is installed in your environment."
        "\n\nAvailable names: {available}"
    )
    E103 = (
        "Found non-serializable Python object in config. Configs should "
        "only include values that can be serialized to JSON. If you need "
        "to pass models or other objects to your component, use a reference "
        "to a registered function or initialize the object in your "
        "component.\n\n{config}"
    )
    E104 = (
        "Can't write to frozen list. Maybe you're trying to modify a computed "
        "property or default function argument?"
    )
