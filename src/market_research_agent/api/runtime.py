"""Lazily-built, shared agent runtime (LLM clients + compiled graph) for the API and UI"""

import threading

from market_research_agent.agent.graph import Dependencies, build_graph, default_dependencies


class Runtime:
    """Holds the agent's dependencies and compiled graph"""

    def __init__(self, deps: Dependencies | None = None):
        self._deps = deps
        self._graph = None
        self._lock = threading.Lock()

    @property
    def deps(self) -> Dependencies:
        with self._lock:
            if self._deps is None:
                self._deps = default_dependencies()
            return self._deps

    @property
    def graph(self):
        deps = self.deps
        with self._lock:
            if self._graph is None:
                # Requests are independent, so no checkpointer: nothing accumulates in memory.
                self._graph = build_graph(deps, checkpointer=False)
            return self._graph
