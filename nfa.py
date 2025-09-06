# Nondeterministic finite automaton

from enum import Enum
from collections import deque
from dataclasses import dataclass
from collections.abc import Iterable
from uuid import uuid4


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
type States = Iterable[State]
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
        return self.transition_function.get(state, dict())

    def read_transition(
        self, state: State, symbol: TransitionSymbol
    ) -> set[State]:
        return self.read_state_map(state).get(symbol, set())

    def epsilon_closure(self, states: States) -> set[State]:
        out = set(states)
        queue = deque(out)
        while queue:
            cur = queue.popleft()
            for s in self.read_transition(cur, EMPTY_STRING):
                if s not in out:
                    out.add(s)
                    queue.append(s)
        return out

    def move_set(self, states: States, symbol: TransitionSymbol) -> set[State]:
        out = set()
        for q in states:
            out.update(self.read_transition(q, symbol))
            out.update(self.read_transition(q, ANY_SYMBOL))
        return out

    def compute(self, input: str) -> ComputationResult:
        nfa_states = self.epsilon_closure({self.start_state})
        for c in input:
            nfa_states = self.move_set(nfa_states, Symbol(c))
            if not nfa_states:
                return ComputationResult.REJECT
            nfa_states = self.epsilon_closure(nfa_states)
        if nfa_states.intersection(self.accept_states):
            return ComputationResult.ACCEPT
        return ComputationResult.REJECT

    def compute_equivalent_DFA(self):
        names: dict[frozenset[State], State] = {}

        def create_name(state: frozenset[State]) -> State:
            state_name = names.get(state, None)
            if state_name is None:
                state_name = '{' + ','.join(state) + '}'
                names[state] = state_name
            return state_name

        dfa_start_state = frozenset(self.epsilon_closure({self.start_state}))
        create_name(dfa_start_state)

        dfa_alphabet = self.alphabet
        try:
            dfa_alphabet.remove(EMPTY_STRING)
        except KeyError:
            pass
        try:
            dfa_alphabet.remove(ANY_SYMBOL)
        except KeyError:
            pass

        sink_state = frozenset({uuid4().hex})
        create_name(sink_state)
        used_sink_state = False

        dfa_transition_function = dict()

        def add_transition(
            state: frozenset[State],
            symbol: Symbol,
            output: frozenset[State]
        ):
            state_name = create_name(state)
            state_transitions = dfa_transition_function.get(state_name, None)
            if state_transitions is None:
                dfa_transition_function[state_name] = dict()
                state_transitions = dfa_transition_function[state_name]
            symbol_transitions = state_transitions.get(symbol, None)
            if symbol_transitions is None:
                state_transitions[symbol] = set()
                symbol_transitions = state_transitions[symbol]
            symbol_transitions.add(create_name(output))

        dfa_accept_states: set[States] = set()
        if dfa_start_state.intersection(self.accept_states):
            dfa_accept_states.add(create_name(dfa_start_state))

        queue = deque([dfa_start_state])
        while queue:
            current = queue.popleft()
            for symbol in dfa_alphabet:
                new_dfa_state = self.move_set(current, symbol)
                new_dfa_state = frozenset(self.epsilon_closure(new_dfa_state))
                if not new_dfa_state:
                    used_sink_state = True
                    add_transition(current, symbol, sink_state)
                elif new_dfa_state not in names:
                    new_state_name = create_name(new_dfa_state)
                    if new_dfa_state.intersection(self.accept_states):
                        dfa_accept_states.add(new_state_name)
                    queue.append(new_dfa_state)
                    add_transition(current, symbol, new_dfa_state)

        if used_sink_state:
            add_transition(sink_state, ANY_SYMBOL, sink_state)
            dfa_alphabet.add(ANY_SYMBOL)

        return NFA(
            states=set(names.values()),
            alphabet=dfa_alphabet,
            transition_function=dfa_transition_function,
            start_state=create_name(dfa_start_state),
            accept_states=dfa_accept_states
        )

    def enumerate_language(self):
        queue = deque([''])
        while True:
            cur = queue.popleft()
            if self.compute(cur) == ComputationResult.ACCEPT:
                print(cur)
            for symbol in self.alphabet:
                match symbol:
                    case Symbol(c):
                        queue.append(cur + c)


if __name__ == '__main__':
    N1 = NFA(
        # Accept strings with even number of a's
        states={'even', 'odd'},
        alphabet={Symbol('a'), Symbol('b')},
        transition_function={
            'even': {
                Symbol('a'): {'odd'},
                Symbol('b'): {'even'},
            },
            'odd': {
                Symbol('a'): {'even'},
                Symbol('b'): {'odd'},
            },
        },
        start_state='even',
        accept_states={'even'}
    )

    N2 = NFA(
        # Accept strings with only a's and no b's
        states={'A', 'B'},
        alphabet={Symbol('a'), Symbol('b')},
        transition_function={
            'A': {
                Symbol('a'): {'A'},
                Symbol('b'): {'B'},
            },
            'B': {
                ANY_SYMBOL: {'B'},
            }
        },
        start_state='A',
        accept_states={'A'}
    )

    N3 = NFA(
        # Accept (a|b)*ab
        states={'0', '1', '2'},
        alphabet={Symbol('a'), Symbol('b')},
        transition_function={
            '0': {
                Symbol('a'): {'0', '1'},
                Symbol('b'): {'0'},
            },
            '1': {
                Symbol('b'): {'2'}
            }
        },
        start_state='0',
        accept_states={'2'}
    )

    N4 = NFA(
        # Accept (a|b)*ab, DFA of N3
        states={'0', '1', '2'},
        alphabet={Symbol('a'), Symbol('b')},
        transition_function={
            '0': {
                Symbol('a'): {'1'},
                Symbol('b'): {'0'},
            },
            '1': {
                Symbol('a'): {'1'},
                Symbol('b'): {'2'},
            },
            '2': {
                Symbol('a'): {'1'},
                Symbol('b'): {'0'},
            }
        },
        start_state='0',
        accept_states={'2'}
    )

    N5 = NFA(
        states={'0', '1', '2', '3', '4', '5', '6', '7', '8'},
        alphabet={Symbol('a'), Symbol('b')},
        transition_function={
            '0': {
                EMPTY_STRING: {'1', '7'},
            },
            '1': {
                EMPTY_STRING: {'2', '4'},
            },
            '2': {
                Symbol('a'): {'3'},
            },
            '3': {
                EMPTY_STRING: {'6'},
            },
            '4': {
                Symbol('b'): {'5'},
            },
            '5': {
                EMPTY_STRING: {'6'},
            },
            '6': {
                EMPTY_STRING: {'1', '7'},
            },
            '7': {
                Symbol('a'): {'8'}
            }
        },
        start_state='0',
        accept_states={'8'}
    )

    N6 = NFA(
        # Accepts abb*a(a|bb*a)*, DFA of N5
        states={'s0', 's1', 's2'},
        alphabet={Symbol('a'), Symbol('b')},
        transition_function={
            's0': {
                Symbol('a'): {'s1'},
                Symbol('b'): {'s2'},
            },
            's1': {
                Symbol('a'): {'s1'},
                Symbol('b'): {'s2'},
            },
            's2': {
                Symbol('a'): {'s1'},
                Symbol('b'): {'s2'},
            }
        },
        start_state='s0',
        accept_states={'s1'}
    )

    N7 = NFA(
        # Accepts (ab)*
        states={'0', '1', '2', '3'},
        alphabet={Symbol('a'), Symbol('b')},
        transition_function={
            '0': {
                Symbol('a'): {'1'},
                Symbol('b'): {'4'},
            },
            '1': {
                Symbol('a'): {'3'},
                Symbol('b'): {'2'},
            },
            '2': {
                Symbol('a'): {'1'},
                Symbol('b'): {'3'},
            },
            '3': {
                ANY_SYMBOL: {'3'}
            }
        },
        start_state='0',
        accept_states={'2'}
    )

    N8 = NFA(
        # Accepts a*b*
        states={'0', '1', '2'},
        alphabet={Symbol('a'), Symbol('b')},
        transition_function={
            '0': {
                Symbol('a'): {'0'},
                Symbol('b'): {'1'},
            },
            '1': {
                Symbol('a'): {'2'},
                Symbol('b'): {'1'},
            },
            '2': {
                ANY_SYMBOL: {'3'}
            }
        },
        start_state='0',
        accept_states={'0', '1'}
    )

    used = N1
    # print(used.compute('bbbb'))
    used.enumerate_language()
