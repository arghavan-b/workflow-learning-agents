"""UI state graph: nodes are distinct GUI states, edges are inferred actions.

For GUI demonstrations a transition graph over *stable UI states* is more useful
than a linear list of temporal video segments: deduplicating states turns a
single linear demo into a reusable app map.

    s_i --a_i--> s_{i+1}

State dedup (``same_state``) ignores transient differences - blinking cursor,
clock, pointer position, notification badges - so two near-identical screenshots
collapse to one node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from demo2skill.video.statediff.cursor import CursorTrack
from demo2skill.video.statediff.inverse_dynamics import StateDiffIDM, TransitionProposer
from demo2skill.video.statediff.state import ScreenState, same_state


@dataclass
class StateNode:
    node_id: int
    label: str
    member_state_indices: List[int] = field(default_factory=list)


@dataclass
class StateEdge:
    src: int
    dst: int
    action_type: str
    target: Optional[str]
    confidence: float

    def label(self) -> str:
        return f"{self.action_type} {self.target}".strip()


class UIStateGraph:
    def __init__(self) -> None:
        self.nodes: List[StateNode] = []
        self.edges: List[StateEdge] = []

    def _node_for(self, state: ScreenState) -> StateNode:
        for node in self.nodes:
            rep = self._states[node.member_state_indices[0]]
            if same_state(rep, state):
                node.member_state_indices.append(state.index)
                return node
        node = StateNode(
            node_id=len(self.nodes),
            label=state.title or state.url or f"state_{state.index}",
            member_state_indices=[state.index],
        )
        self.nodes.append(node)
        return node

    @classmethod
    def build(cls, states: List[ScreenState], cursor: Optional[CursorTrack] = None) -> "UIStateGraph":
        graph = cls()
        graph._states = {s.index: s for s in states}
        idm = StateDiffIDM(cursor or CursorTrack.empty())

        for tr in TransitionProposer().propose(states):
            action = idm.infer(tr.before, tr.after)
            if action.action_type == "noop":
                continue
            src = graph._node_for(tr.before)
            dst = graph._node_for(tr.after)
            graph.edges.append(StateEdge(
                src=src.node_id, dst=dst.node_id,
                action_type=action.action_type,
                target=action.target.display() if action.target else None,
                confidence=action.confidence,
            ))
        return graph

    def to_dict(self) -> dict:
        return {
            "nodes": [{"id": n.node_id, "label": n.label,
                       "states": n.member_state_indices} for n in self.nodes],
            "edges": [{"src": e.src, "dst": e.dst, "action": e.label(),
                       "confidence": e.confidence} for e in self.edges],
        }

    def ascii(self) -> str:
        lines = []
        for e in self.edges:
            s = self.nodes[e.src].label
            d = self.nodes[e.dst].label
            lines.append(f"[{s}] --{e.label()}--> [{d}]")
        return "\n".join(lines)
