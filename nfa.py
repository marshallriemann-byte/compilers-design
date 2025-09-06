# Nondeterministic finite automaton

from enum import Enum
from collections import deque
from dataclasses import dataclass


class TransitionSymbol:
    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))


class EmptyString(TransitionSymbol):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance


class AnySymbol(TransitionSymbol):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance


@dataclass(frozen=True)
class Symbol(TransitionSymbol):
    char: str


EMPTY_STRING = EmptyString()
ANY_SYMBOL = AnySymbol()
type Alphabet = set[Alphabet]
type State = str
type States = set[State]
type StateMap = dict[TransitionSymbol, States]
type TransitionFunction = dict[State, StateMap]


class ComputationResult(Enum):
    ACCEPT = 0
    REJECT = 1


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
        return self.transition_function.get(state, StateMap())

    def read_transition(
        self, state: State, symbol: TransitionSymbol
    ) -> States:
        return self.read_state_map(state).get(symbol, States())

    def epsilon_closure(self, states: States) -> States:
        out = set(states)
        queue = deque(out)
        while queue:
            cur = queue.popleft()
            for s in self.read_transition(cur, EMPTY_STRING):
                if s not in out:
                    out.add(s)
                    queue.append(s)
        return out

    def move_set(self, states: States, symbol: TransitionSymbol) -> States:
        out = set()
        for q in states:
            out.update(self.read_transition(q, symbol))
            out.update(self.read_transition(q, ANY_SYMBOL))
        return out
