"""DSPy language model configuration module.

Provides functions to configure and retrieve the global DSPy LM instance used
throughout the classification system.
"""


import dspy

_lm: dspy.LM | None = None


def configure(lm: dspy.LM) -> None:
    global _lm
    _lm = lm
    dspy.configure(lm=lm)


def get_lm() -> dspy.LM | None:
    return _lm
