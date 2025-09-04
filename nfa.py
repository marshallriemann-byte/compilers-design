# Nondeterministic finite automaton

from collections import deque
from dataclasses import dataclass


class EmptyString:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        cls._instance


class AnySymbol:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        cls._instance


@dataclass
class Symbol:
    char: str


AlphabetSymbol = EmptyString | AnySymbol | Symbol
EMPTY_STRING = EmptyString()


class NFA:
    def __init__(
        self,
        states: set[str],
        alphabet: set[str],
        transition_function: dict[str, dict[AlphabetSymbol, set[str]]],
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
