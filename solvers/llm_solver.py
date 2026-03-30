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

COORDINATE SYSTEM:
- The grid is shown with row 0 at the TOP and column 0 at the LEFT.
- UP moves you to a LOWER row number (visually upward on screen).
- DOWN moves you to a HIGHER row number (visually downward on screen).
- LEFT moves you to a LOWER column number (visually left on screen).
- RIGHT moves you to a HIGHER column number (visually right on screen).

TILE LEGEND:
- P = your position
- # = wall (impassable — if you try to move into a wall, you stay in place!)
- . = floor (safe to walk on)
- X = trap (damages you -25 HP, avoid if possible)
- K = key (collect ALL keys before you can open doors)
- D = door (requires all keys collected to pass through)
- G = goal (reach this to win!)
- E = enemy (patrols around, damages you -30 HP on contact)

STRATEGY:
1. Look at the tiles ADJACENT to P (up, down, left, right).
2. Only move toward a tile that is NOT a wall (#).
3. If your last move did not change your position, you hit a wall — try a DIFFERENT direction!
4. Collect all keys (K) before heading to the door (D), then reach the goal (G).
5. Avoid traps (X) and enemies (E) when possible.

Respond with ONLY a JSON object: {"action": "UP"} or {"action": "DOWN"} etc.
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
        self._prev_pos = None
        self._last_action = None
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
        elif self.provider == "groq":
            try:
                import openai
                import os
                self.client = openai.OpenAI(
                    api_key=os.environ.get("GROQ_API_KEY"),
                    base_url="https://api.groq.com/openai/v1",
                )
                self.model = self.model or "llama-3.3-70b-versatile"
            except ImportError:
                raise ImportError("pip install openai")
        elif self.provider == "ollama":
            try:
                import openai
                self.client = openai.OpenAI(
                    api_key="ollama",
                    base_url="http://localhost:11434/v1",
                )
                self.model = self.model or "llama3.2"
            except ImportError:
                raise ImportError("pip install openai")
        else:
            raise ValueError(f"Unknown provider: {self.provider}. Use 'claude', 'openai', 'groq', or 'ollama'.")

    def _ask(self, state_text: str, feedback: str = "") -> str:
        """Send the current state to the LLM and get an action."""
        user_msg = f"Current game state:\n\n{state_text}"
        if feedback:
            user_msg += f"\n\n{feedback}"
        user_msg += "\n\nWhat is your next move? Respond with JSON only."

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
        self._prev_pos = None
        self._last_action = None
        actions = []
        current = state.copy()

        for step in range(LLM_MAX_STEPS):
            if current.done:
                break

            state_text = current.to_text()
            feedback = self._make_feedback(current)
            try:
                response = self._ask(state_text, feedback)
                action = self._parse_action(response)
            except Exception as e:
                print(f"LLM error at step {step}: {e}")
                action = WAIT

            self._prev_pos = current.player_pos
            self._last_action = action
            actions.append(action)
            current, reward, _ = current.step(action)
            status = "WON!" if current.won else ("DEAD" if current.done else "")
            print(f"    Step {step+1}: {ACTION_NAMES[action]:5s} | HP: {current.health} | Keys: {len(current.keys_collected)}/{current.total_keys} {status}")

        return actions

    def solve_step(self, state: GameState) -> int:
        """Single step — useful for real-time visualization."""
        state_text = state.to_text()
        feedback = self._make_feedback(state)
        try:
            response = self._ask(state_text, feedback)
            action = self._parse_action(response)
        except Exception as e:
            print(f"LLM error: {e}")
            action = WAIT

        self._prev_pos = state.player_pos
        self._last_action = action
        return action

    def _make_feedback(self, state: GameState) -> str:
        """Generate feedback about the previous move."""
        if self._prev_pos is None or self._last_action is None:
            return ""

        parts = []
        if state.player_pos == self._prev_pos:
            parts.append(
                f"WARNING: Your last move ({ACTION_NAMES[self._last_action]}) "
                f"FAILED — you hit a wall and stayed at the same position! "
                f"Try a DIFFERENT direction."
            )
        else:
            parts.append(
                f"Your last move ({ACTION_NAMES[self._last_action]}) succeeded: "
                f"moved from row {self._prev_pos[0]},col {self._prev_pos[1]} "
                f"to row {state.player_pos[0]},col {state.player_pos[1]}."
            )
        return " ".join(parts)

    def reset(self) -> None:
        self.conversation_history = []
        self._prev_pos = None
        self._last_action = None
