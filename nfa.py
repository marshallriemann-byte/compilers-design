# Nondeterministic finite automaton

from enum import Enum
from collections import deque
from dataclasses import dataclass
from collections.abc import Iterable, Sequence
from uuid import uuid4
from typing import Self
from copy import deepcopy


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

    def compute_equivalent_DFA(self) -> Self:
        names: dict[frozenset[State], State] = dict()

        def create_name(state: frozenset[State]) -> State:
            state_name = names.get(state, None)
            if state_name is None:
                state_name = '{' + ', '.join(state) + '}'
                names[state] = state_name
            return state_name

        dfa_start_state = frozenset(self.epsilon_closure({self.start_state}))

        dfa_alphabet = self.alphabet
        try:
            dfa_alphabet.remove(EMPTY_STRING)
        except KeyError:
            pass

        sink_state = frozenset({uuid4().hex})
        used_sink_state = False

        dfa_transition_function = dict()

        def add_transition(
            state: frozenset[State],
            symbol: TransitionSymbol,
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

        queue = {dfa_start_state}
        while queue:
            current = queue.pop()
            if current.intersection(self.accept_states):
                dfa_accept_states.add(create_name(current))
            for symbol in dfa_alphabet:
                new_dfa_state = self.move_set(current, symbol)
                new_dfa_state = frozenset(self.epsilon_closure(new_dfa_state))
                if not new_dfa_state:
                    used_sink_state = True
                    add_transition(current, symbol, sink_state)
                else:
                    add_transition(current, symbol, new_dfa_state)
                    if create_name(new_dfa_state) not in dfa_transition_function:
                        queue.add(new_dfa_state)

        if used_sink_state:
            add_transition(sink_state, ANY_SYMBOL, sink_state)

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

    def kleene_star(nfa: Self) -> Self:
        star_nfa_start_state = uuid4().hex
        star_nfa = deepcopy(nfa)
        star_nfa.states.add(star_nfa_start_state)
        star_nfa.accept_states.add(star_nfa_start_state)
        for q in star_nfa.accept_states:
            q_state_map = star_nfa.transition_function.get(q, None)
            if not q_state_map:
                star_nfa.transition_function[q] = dict()
                q_state_map = star_nfa.transition_function[q]
            try:
                q_state_map[EMPTY_STRING].add(star_nfa.start_state)
            except KeyError:
                q_state_map[EMPTY_STRING] = {star_nfa.start_state}
        star_nfa.start_state = star_nfa_start_state
        return star_nfa

    def concatenate(automata: Sequence[Self]) -> Self:
        def rename_state(nfa_index, state_name):
            return f'(NFA{nfa_index}, {state_name})'
        states: set[States] = set()
        alphabet: Alphabet = set()
        transition_function: TransitionFunction = dict()
        # Rename states
        for (index, nfa) in enumerate(automata):
            states.update({rename_state(index, q) for q in nfa.states})
            alphabet.update(nfa.alphabet)
            for (q, q_map) in nfa.transition_function.items():
                transition_function[rename_state(index, q)] = {
                    symbol: {
                        rename_state(index, r) for r in symbol_map
                    }
                    for (symbol, symbol_map) in q_map.items()
                }
            if index+1 < len(automata):
                for q in nfa.accept_states:
                    q_name = rename_state(index, q)
                    q_map = transition_function.get(q, None)
                    if not q_map:
                        transition_function[q_name] = dict()
                        q_map = transition_function[q_name]
                    next_nfa_start_state = rename_state(
                        index+1,
                        automata[index+1].start_state
                    )
                    try:
                        q_map[EMPTY_STRING].update(next_nfa_start_state)
                    except KeyError:
                        q_map[EMPTY_STRING] = {next_nfa_start_state}
        start_state = rename_state(0, automata[0].start_state)
        accept_states = {
            rename_state(len(automata)-1, q)
            for q in automata[-1].accept_states
        }
        return NFA(
            states,
            alphabet,
            transition_function,
            start_state,
            accept_states
        )


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
        # Accepts aa*|(b|aa*b)(b|aa*b)*aa*, DFA of N5
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

    N9 = NFA(
        # Accepts a|b
        states={'0', '1', '2', '3', '4'},
        alphabet={Symbol('a'), Symbol('b')},
        transition_function={
            '0': {
                EMPTY_STRING: {'1', '2'}
            },
            '1': {
                Symbol('a'): {'3'},
            },
            '2': {
                Symbol('b'): {'4'}
            }
        },
        start_state='0',
        accept_states={'3', '4'}
    )

    N10 = NFA(
        # DFA of N9
        states={'0', '1', '2', '3'},
        alphabet={Symbol('a'), Symbol('b')},
        transition_function={
            '0': {
                Symbol('a'): {'1'},
                Symbol('b'): {'2'},
            },
            '1': {
                ANY_SYMBOL: {'3'}
            },
            '2': {
                ANY_SYMBOL: {'3'}
            },
            '3': {
                ANY_SYMBOL: {'3'}
            }
        },
        start_state='0',
        accept_states={'1', '2'}
    )

    N11 = NFA(
        # Accepts b*ab*a(a|bb*a)*bb*
        states={'0', '1', '2', '3'},
        alphabet={Symbol('a'), Symbol('b')},
        transition_function={
            '0': {
                Symbol('a'): {'0', '1'},
                Symbol('b'): {'0'},
            },
            '1': {
                Symbol('a'): {'1', '2'},
                Symbol('b'): {'1'},
            },
            '2': {
                EMPTY_STRING: {'0'},
                Symbol('a'): {'2'},
                Symbol('b'): {'2', '3'},
            },
        },
        start_state='0',
        accept_states={'3'}
    )

    N12 = NFA(
        # DFA of N11
        states={'0', '1', '2', '3'},
        alphabet={Symbol('a'), Symbol('b')},
        transition_function={
            '0': {
                Symbol('a'): {'1'},
                Symbol('b'): {'0'},
            },
            '1': {
                Symbol('a'): {'2'},
                Symbol('b'): {'1'},
            },
            '2': {
                Symbol('a'): {'2'},
                Symbol('b'): {'3'},
            },
            '3': {
                Symbol('a'): {'2'},
                Symbol('b'): {'3'},
            }
        },
        start_state='0',
        accept_states={'3'}
    )

    used = NFA.concatenate([N10, N10])
    print(used.enumerate_language())
    # used.enumerate_language()
