"""
agent/__init__.py
-----------------
Public interface for the TechVest Recruitment Agent.

Usage
-----
    from agent import recruitment_graph, make_config, initial_state
    from agent import AgentState

    state  = initial_state(jd_text, candidates)
    config = make_config(thread_id="session-001")

    # Run up to the human approval interrupt
    result = recruitment_graph.invoke(state, config=config)

    # Resume after human sets approval
    result = recruitment_graph.invoke(
        {"human_approved": True},
        config=config,
    )
"""

from agent.graph import GRAPH_CONFIG, RECURSION_LIMIT, build_graph, make_config, recruitment_graph
from agent.state import AgentState, initial_state

__all__ = [
    # Graph
    "recruitment_graph",
    "build_graph",
    "make_config",
    "GRAPH_CONFIG",
    "RECURSION_LIMIT",
    # State
    "AgentState",
    "initial_state",
]
