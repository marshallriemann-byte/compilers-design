# Nondeterministic finite automaton

from collections import deque

EMPTY_STRING = ''


class NFA:
    def __init__(
        self,
        states: set[str],
        alphabet: set[str],
        transition_function: dict[str, dict[str, set[str]]],
        start_state: str,
        accept_states: set[str]
    ):
        self.states = states
        self.alphabet = alphabet
        self.transition_function = transition_function
        self.start_state = start_state
        self.accept_states = accept_states

    def epsilon_closure(self, states: set[str]) -> set[str]:
        out = set(states)
        queue = deque(out)
        while queue:
            cur = queue.popleft()
            for s in self.transition_function[cur][EMPTY_STRING]:
                if s not in out:
                    out.add(s)
                    queue.append(s)
        out
