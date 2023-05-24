"""
This module provides a helper function that simplyfies the use of quantity
generators in liesel/goose.
"""

from collections.abc import Callable
from typing import ClassVar, NamedTuple

from liesel.goose.epoch import EpochState
from liesel.goose.types import KeyArray, ModelInterface, ModelState, Position
from liesel.option import Option

GenQuantCallable = Callable[
    [KeyArray, ModelState, ModelInterface, EpochState], Position
]


class GeneratedPosition(NamedTuple):
    error_code: int
    position: Position


def quantgen(f: GenQuantCallable, identifier: str | None = None):
    """
    wraps a function that generates a quantity based on model state and prng key
    such that it is accepted by the engine as a instance of the
    `liesel.goose.types.QuantityGenerator` protocol.
    """

    class Generator:
        error_book: ClassVar[dict[int, str]] = {}
        identifier: str = ""

        def __init__(self, f: GenQuantCallable, identifier: str):
            self.identifier = identifier
            self._f = f
            self._model: Option[ModelInterface] = Option(None)

        def set_model(self, model: ModelInterface):
            self._model = Option(model)

        def has_model(self):
            return self._model.is_some()

        def generate(
            self,
            prng_key: KeyArray,
            model_state: ModelState,
            epoch: EpochState,
        ) -> GeneratedPosition:
            mod_interface = self._model.unwrap()
            pos = self._f(prng_key, model_state, mod_interface, epoch)
            return GeneratedPosition(error_code=0, position=pos)

    return Generator(f, identifier if identifier is not None else "")
