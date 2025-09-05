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
type Alphabet = set[Alphabet]
type State = str
type States = set[State]
type StateMap = dict[AlphabetSymbol, States]
type TransitionFunction = dict[State, StateMap]


class NFA:
    def __init__(
        self,
        states: States,
        alphabet: Alphabet,
        transition_function: TransitionFunction,
        start_state: State,
        accept_states: States
    ):
        self.states = states
        self.alphabet = alphabet
        self.transition_function = transition_function
        self.start_state = start_state
        self.accept_states = accept_states

    def read_state_map(self, state: State) -> StateMap:
        self.transition_function.get(state, StateMap())

    def read_transition(
            self, state: State, symbol: AlphabetSymbol
    ) -> States:
        self.read_state_map(state).get(symbol, set())

    def epsilon_closure(self, states: States) -> States:
        out = set(states)
        queue = deque(out)
        while queue:
            cur = queue.popleft()
            for s in self.transition_function[cur][EMPTY_STRING]:
                if s not in out:
                    out.add(s)
                    queue.append(s)
        out

    def move_set(self, states: States, symbol: AlphabetSymbol) -> States:
        out = set()
        for q in states:
            out.union(self.read_transition(q, symbol))
            out.union(self.read_transition(q, ANY_SYMBOL))
        out
