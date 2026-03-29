"""LLM solver — uses Claude or ChatGPT to reason about the puzzle."""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from config import UP, DOWN, LEFT, RIGHT, WAIT, ACTION_NAMES, LLM_MAX_STEPS
from game.engine import GameState
from solvers.base import BaseSolver

SYSTEM_PROMPT = """\
You are an AI agent solving a grid-based puzzle game. You see the grid and must choose actions to reach the goal (G).

Rules:
- P = your position. Move with: UP, DOWN, LEFT, RIGHT, WAIT
- # = walls (impassable)
- . = floor (safe)
- X = trap (damages you, avoid if possible)
- K = key (collect all keys before you can open doors)
- D = door (requires all keys collected to pass through)
- G = goal (reach this to win)
- E = enemy (moves around, damages you on contact, avoid)

Strategy tips:
- Collect all keys (K) before heading to the door (D)
- Avoid traps (X) and enemies (E) when possible
- Find the shortest safe path to the goal (G)

Respond with ONLY a JSON object: {"action": "UP"} or {"action": "DOWN"} etc.
Think about the best move, then respond with the JSON only.
"""


class LLMSolver(BaseSolver):
    """LLM-based solver supporting Claude and ChatGPT."""

    name = "LLM"
    color = "llm"

    def __init__(self, provider: str = "claude", model: Optional[str] = None):
        """
        Args:
            provider: "claude" or "openai"
            model: specific model name (defaults to latest)
        """
        self.provider = provider.lower()
        self.model = model
        self.client = None
        self.conversation_history: list[dict] = []
        self._init_client()

    def _init_client(self) -> None:
        if self.provider == "claude":
            try:
                import anthropic
                self.client = anthropic.Anthropic()
                self.model = self.model or "claude-sonnet-4-20250514"
            except ImportError:
                raise ImportError("pip install anthropic")
        elif self.provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI()
                self.model = self.model or "gpt-4o"
            except ImportError:
                raise ImportError("pip install openai")
        else:
            raise ValueError(f"Unknown provider: {self.provider}. Use 'claude' or 'openai'.")

    def _ask(self, state_text: str) -> str:
        """Send the current state to the LLM and get an action."""
        user_msg = f"Current game state:\n\n{state_text}\n\nWhat is your next move? Respond with JSON only."

        if self.provider == "claude":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=150,
                system=SYSTEM_PROMPT,
                messages=self.conversation_history + [{"role": "user", "content": user_msg}],
            )
            reply = response.content[0].text
        else:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": user_msg})
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=150,
                temperature=0.2,
            )
            reply = response.choices[0].message.content

        # Keep conversation history for context (last 10 exchanges)
        self.conversation_history.append({"role": "user", "content": user_msg})
        self.conversation_history.append({"role": "assistant", "content": reply})
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return reply

    def _parse_action(self, response: str) -> int:
        """Extract action from LLM response."""
        action_map = {"UP": UP, "DOWN": DOWN, "LEFT": LEFT, "RIGHT": RIGHT, "WAIT": WAIT}

        # Try JSON parse
        try:
            # Find JSON in response
            match = re.search(r'\{[^}]+\}', response)
            if match:
                data = json.loads(match.group())
                action_str = data.get("action", "WAIT").upper()
                return action_map.get(action_str, WAIT)
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: look for action keywords in text
        response_upper = response.upper()
        for name, action in action_map.items():
            if name in response_upper:
                return action

        return WAIT

    def solve(self, state: GameState) -> list[int]:
        """Solve by repeatedly asking the LLM for the next move."""
        self.conversation_history = []
        actions = []
        current = state.copy()

        for step in range(LLM_MAX_STEPS):
            if current.done:
                break

            state_text = current.to_text()
            try:
                response = self._ask(state_text)
                action = self._parse_action(response)
            except Exception as e:
                print(f"LLM error at step {step}: {e}")
                action = WAIT

            actions.append(action)
            current, _, _ = current.step(action)

        return actions

    def solve_step(self, state: GameState) -> int:
        """Single step — useful for real-time visualization."""
        state_text = state.to_text()
        try:
            response = self._ask(state_text)
            return self._parse_action(response)
        except Exception as e:
            print(f"LLM error: {e}")
            return WAIT

    def reset(self) -> None:
        self.conversation_history = []
