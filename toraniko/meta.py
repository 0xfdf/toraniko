"""Meta functions used by Toraniko for informing users of impending library changes."""

from functools import wraps
import warnings


def deprecated(version_for_removal: str | None = None, instructions: str | None = None):
    """Alert library users a public function deprecated and due to be removed in an upcoming release.

    Deprecated functions will be decorated with this one. It's not strictly necessary but a good best practice
    to also include `version_for_removal` and further `instructions`. This shouldn't be used unless it's going to
    be removed in the next major release.

    Parameters
    ----------
    version_for_removal: str first version upon which the deprecated function will no longer be available
    instructions: str message to tell users how to best migrate their use case from the deprecated function
    """

    def decorator(func):
        wraps(func)

        def wrapper(*args, **kwargs):
            base_message = (
                f"Function {func.__name__} is deprecated and will be removed "
                f"in {version_for_removal if version_for_removal is not None else 'an upcoming release.'}"
            )
            if instructions:
                full_message = f"{base_message} {instructions}"
            else:
                full_message = base_message

            warnings.warn(full_message, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    # This pattern allows the decorator to be used with or without parentheses
    return decorator if callable(instructions) else lambda func: decorator(func)


def breaking_change(version_for_change: str, instructions: str | None = None):
    """Alert library users a public function is due for a breaking API change in the next major release.

    Unlike the `deprecated` decorator, `version_for_change` must be included as an argument here. But further
    `instructions` are optional.

    Parameters
    ----------
    version_for_change: str first version upon which the function will have a breaking change
    instructions: str message to tell users how to best migrate their use case from the deprecated function
    """

    def decorator(func):
        wraps(func)

        def wrapper(*args, **kwargs):
            base_message = (
                f"Function {func.__name__} will have a breaking change to its arguments in {version_for_change}."
            )
            if instructions:
                full_message = f"{base_message} {instructions}"
            else:
                full_message = base_message

            warnings.warn(full_message, category=FutureWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def unstable():
    """Alert library users a public function is considered unstable and may or may not be removed or break at any time."""

    def decorator(func):
        wraps(func)

        def wrapper(*args, **kwargs):
            message = (
                f"Function {func.__name__} is considered unstable! It may or may not be removed, or break, at any time."
            )

            warnings.warn(message, category=Warning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return lambda func: decorator(func)
