"""Base solver interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from game.engine import GameState


class BaseSolver(ABC):
    """All solvers implement this interface."""

    name: str = "base"
    color: str = "text"

    @abstractmethod
    def solve(self, state: GameState) -> list[int]:
        """Return a list of actions to solve the level, or best-effort attempt."""

    def reset(self) -> None:
        """Reset any internal state between levels."""
