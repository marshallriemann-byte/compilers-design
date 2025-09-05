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
type State = str


class NFA:
    def __init__(
        self,
        states: set[State],
        alphabet: set[AlphabetSymbol],
        transition_function: dict[State, dict[AlphabetSymbol, set[State]]],
        start_state: State,
        accept_states: set[State]
    ):
        self.states = states
        self.alphabet = alphabet
        self.transition_function = transition_function
        self.start_state = start_state
        self.accept_states = accept_states

    def epsilon_closure(self, states: set[State]) -> set[State]:
        out = set(states)
        queue = deque(out)
        while queue:
            cur = queue.popleft()
            for s in self.transition_function[cur][EMPTY_STRING]:
                if s not in out:
                    out.add(s)
                    queue.append(s)
        out
