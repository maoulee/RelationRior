from __future__ import annotations

from typing import Dict

from subgraph_kgqa.rl.plugin import PhaseHintGenerator


class OneShotStageHintManager:
    """Emit normal stage hints once, and suppress repeated repair hints."""

    @staticmethod
    def next_hint(state: Dict) -> str:
        key = OneShotStageHintManager._hint_key(state)
        if not key:
            return ""

        if key.startswith("stage:"):
            emitted = state.setdefault("_emitted_stage_hint_keys", set())
            if key in emitted:
                return ""
            hint = PhaseHintGenerator.generate(state)
            if hint:
                emitted.add(key)
                state["hint_history"] = [*state.get("hint_history", []), key]
            return hint

        last_key = state.get("_last_hint_key")
        if last_key == key:
            return ""

        hint = PhaseHintGenerator.generate(state)
        if hint:
            state["_last_hint_key"] = key
            state["hint_history"] = [*state.get("hint_history", []), key]
        return hint

    @staticmethod
    def clear_repair_tracking(state: Dict) -> None:
        state["_last_hint_key"] = None

    @staticmethod
    def _hint_key(state: Dict) -> str:
        repair_mode = state.get("repair_mode")
        if repair_mode:
            return f"repair:{repair_mode}"
        stage = PhaseHintGenerator.detect_stage(state)
        return f"stage:{stage}"
