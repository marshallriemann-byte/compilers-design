# Nondeterministic finite automaton

from enum import Enum
from collections import deque
from collections.abc import Sequence
from uuid import uuid4
from typing import Self
from copy import deepcopy


type Symbol = str
type Alphabet = set[Symbol]
type State = str
type States = set[State]
type StateMap = dict[Symbol, States]
type TransitionFunction = dict[State, StateMap]


EMPTY_STRING = ''


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
        self.states = set(states)
        self.alphabet = set(alphabet)
        self.transition_function = deepcopy(transition_function)
        for (q, q_map) in self.transition_function.items():
            self.states.add(q)
            for (symbol, symbol_set) in q_map.items():
                if symbol != EMPTY_STRING:
                    self.alphabet.add(symbol)
                self.states.update(symbol_set)
        self.start_state = start_state
        self.states.add(self.start_state)
        self.accept_states = set(accept_states)
        self.states.update(self.accept_states)

    def read_state_map(self, state: State) -> StateMap:
        return self.transition_function.get(state, dict())

    def read_transition(
        self, state: State, symbol: Symbol
    ) -> States:
        return self.read_state_map(state).get(symbol, set())

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

    def move_set(self, states: States, symbol: Symbol) -> States:
        out = set()
        for q in states:
            out.update(self.read_transition(q, symbol))
        return out

    def compute(self, input: str) -> ComputationResult:
        nfa_states = self.epsilon_closure({self.start_state})
        for c in input:
            nfa_states = self.move_set(nfa_states, c)
            if not nfa_states:
                return ComputationResult.REJECT
            nfa_states = self.epsilon_closure(nfa_states)
        if nfa_states.intersection(self.accept_states):
            return ComputationResult.ACCEPT
        return ComputationResult.REJECT

    def enumerate_language(self):
        queue = deque([''])
        while True:
            cur = queue.popleft()
            if self.compute(cur) == ComputationResult.ACCEPT:
                print(cur)
            for symbol in self.alphabet:
                queue.append(cur + symbol)

    def compute_equivalent_DFA(self) -> Self:
        names: dict[States, State] = dict()

        def create_name(state: States) -> State:
            return names.setdefault(
                state,
                '{' + ', '.join(state) + '}'
            )

        dfa_start_state = frozenset(self.epsilon_closure({self.start_state}))

        dfa_alphabet = {s for s in self.alphabet if s != EMPTY_STRING}

        sink_state = frozenset({f'(SINK, {uuid4().hex})'})
        used_sink_state = False

        dfa_transition_function = dict()

        def add_transition(
            state: States,
            symbol: Symbol,
            output: States
        ):
            dfa_transition_function.setdefault(
                create_name(state), dict()
            ).setdefault(
                symbol, set()
            ).add(
                create_name(output)
            )

        dfa_accept_states: States = set()

        queue = deque([dfa_start_state])
        while queue:
            current = queue.popleft()
            if current.intersection(self.accept_states):
                dfa_accept_states.add(create_name(current))
            for symbol in dfa_alphabet:
                new_dfa_state = self.move_set(current, symbol)
                new_dfa_state = frozenset(self.epsilon_closure(new_dfa_state))
                if not new_dfa_state:
                    add_transition(current, symbol, sink_state)
                    used_sink_state = True
                else:
                    add_transition(current, symbol, new_dfa_state)
                    new_name = create_name(new_dfa_state)
                    if new_name not in dfa_transition_function:
                        queue.append(new_dfa_state)

        if used_sink_state:
            sink_state_name = create_name(sink_state)
            dfa_transition_function[sink_state_name] = {
                symbol: {sink_state_name}
                for symbol in dfa_alphabet
            }

        return NFA(
            states=set(names.values()),
            alphabet=dfa_alphabet,
            transition_function=dfa_transition_function,
            start_state=create_name(dfa_start_state),
            accept_states=dfa_accept_states
        )

    def compute_minimized_DFA(self):
        partitions = set()
        X = frozenset(self.accept_states)
        if X:
            partitions.add(frozenset(X))
        Y = frozenset(self.states - self.accept_states)
        if Y:
            partitions.add(frozenset(Y))
        queue = set(partitions)
        while queue:
            A = queue.pop()
            for c in self.alphabet:
                X = set()
                for q in self.states:
                    S = self.move_set({q}, c)
                    if not S:
                        raise ValueError(
                            f'DFA state {q} has no trasition on symbol {c}'
                        )
                    elif S.issubset(A):
                        X.add(q)
                if not X:
                    continue
                for P in partitions.copy():
                    S = frozenset(P.intersection(X))
                    R = frozenset(P - X)
                    if S and R:
                        partitions.remove(P)
                        partitions.update([S, R])
                        if P in queue:
                            queue.remove(P)
                            queue.update([S, R])
                        elif len(S) < len(R):
                            queue.add(S)
                        else:
                            queue.add(R)

        names: dict[States, State] = dict()

        def create_name(state: States) -> State:
            return names.setdefault(
                state,
                '{' + ', '.join(state) + '}'
            )

        states: States = set()
        alphabet = set(self.alphabet)
        transition_function: TransitionFunction = dict()
        accept_states: States = set()
        states_partitions_map = {
            q: create_name(P)
            for P in partitions
            for q in P
        }
        for P in partitions:
            P_name = create_name(P)
            states.add(P_name)
            if P.intersection(self.accept_states):
                accept_states.add(P_name)
            transition_function[P_name] = {
                c: {states_partitions_map[self.move_set(P, c).pop()]}
            }
        start_state = states_partitions_map[self.start_state]
        return NFA(
            states,
            alphabet,
            transition_function,
            start_state,
            accept_states
        )

    def rename_states(self) -> Self:
        counter = 0
        names: dict[State, int] = dict()
        for q in self.states:
            names[q] = str(counter)
            counter += 1
        self.states = {names[q] for q in self.states}
        self.transition_function = {
            names[q]: {
                c: {names[r] for r in c_set}
                for (c, c_set) in q_map.items()
            }
            for (q, q_map) in self.transition_function.items()
        }
        self.start_state = names[self.start_state]
        self.accept_states = {names[q] for q in self.accept_states}
        return self

    def __mul__(self, other: Self) -> Self:
        return NFA.concatenate([self, other])

    def __or__(self, other: Self) -> Self:
        return NFA.union([self, other])

    def __invert__(self) -> Self:
        return NFA.kleene_star(self)

    @staticmethod
    def kleene_star(nfa: Self) -> Self:
        star_nfa_start_state = f'(STAR, {uuid4().hex})'
        star_nfa = deepcopy(nfa)
        star_nfa.states.add(star_nfa_start_state)
        star_nfa.accept_states.add(star_nfa_start_state)
        for q in star_nfa.accept_states:
            star_nfa.transition_function.setdefault(
                q, dict()
            ).setdefault(
                EMPTY_STRING, set()
            ).add(
                star_nfa.start_state
            )
        star_nfa.start_state = star_nfa_start_state
        return star_nfa

    @staticmethod
    def concatenate(automata: Sequence[Self]) -> Self:
        def rename_state(nfa_index, state_name):
            return f'(NFA{nfa_index}, {state_name})'
        states: States = set()
        alphabet: Alphabet = set()
        transition_function: TransitionFunction = dict()
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
                next_nfa_start_state = rename_state(
                    index+1,
                    automata[index+1].start_state
                )
                for q in nfa.accept_states:
                    transition_function.setdefault(
                        rename_state(index, q), dict()
                    ).setdefault(
                        EMPTY_STRING, set()
                    ).add(
                        next_nfa_start_state
                    )
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

    @staticmethod
    def union(automata: Sequence[Self]) -> Self:
        def rename_state(nfa_index, state_name):
            return f'(NFA{nfa_index}, {state_name})'
        states: States = set()
        alphabet: Alphabet = set()
        transition_function: TransitionFunction = dict()
        start_state = f'(UNION, {uuid4().hex})'
        states.add(start_state)
        transition_function[start_state] = {
            EMPTY_STRING: set()
        }
        start_state_set: States =\
            transition_function[start_state][EMPTY_STRING]
        accept_states: States = set()
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
            start_state_set.add(
                rename_state(index, nfa.start_state)
            )
            accept_states.update({
                rename_state(index, q)
                for q in nfa.accept_states
            })
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
        alphabet={'a', 'b'},
        transition_function={
            'even': {
                'a': {'odd'},
                'b': {'even'},
            },
            'odd': {
                'a': {'even'},
                'b': {'odd'},
            },
        },
        start_state='even',
        accept_states={'even'}
    )

    N2 = NFA(
        # Accept strings with only a's and no b's
        states={'A', 'B'},
        alphabet={'a', 'b'},
        transition_function={
            'A': {
                'a': {'A'},
                'b': {'B'},
            },
            'B': {
                'a': {'B'},
                'b': {'B'},
            }
        },
        start_state='A',
        accept_states={'A'}
    )

    N3 = NFA(
        # Accept (a|b)*ab
        states={'0', '1', '2'},
        alphabet={'a', 'b'},
        transition_function={
            '0': {
                'a': {'0', '1'},
                'b': {'0'},
            },
            '1': {
                'b': {'2'}
            }
        },
        start_state='0',
        accept_states={'2'}
    )

    N4 = NFA(
        # Accept (a|b)*ab, DFA of N3
        states={'0', '1', '2'},
        alphabet={'a', 'b'},
        transition_function={
            '0': {
                'a': {'1'},
                'b': {'0'},
            },
            '1': {
                'a': {'1'},
                'b': {'2'},
            },
            '2': {
                'a': {'1'},
                'b': {'0'},
            }
        },
        start_state='0',
        accept_states={'2'}
    )

    N5 = NFA(
        states={'0', '1', '2', '3', '4', '5', '6', '7', '8'},
        alphabet={'a', 'b'},
        transition_function={
            '0': {
                EMPTY_STRING: {'1', '7'},
            },
            '1': {
                EMPTY_STRING: {'2', '4'},
            },
            '2': {
                'a': {'3'},
            },
            '3': {
                EMPTY_STRING: {'6'},
            },
            '4': {
                'b': {'5'},
            },
            '5': {
                EMPTY_STRING: {'6'},
            },
            '6': {
                EMPTY_STRING: {'1', '7'},
            },
            '7': {
                'a': {'8'}
            }
        },
        start_state='0',
        accept_states={'8'}
    )

    N6 = NFA(
        # Accepts aa*|(b|aa*b)(b|aa*b)*aa*, DFA of N5
        states={'s0', 's1', 's2'},
        alphabet={'a', 'b'},
        transition_function={
            's0': {
                'a': {'s1'},
                'b': {'s2'},
            },
            's1': {
                'a': {'s1'},
                'b': {'s2'},
            },
            's2': {
                'a': {'s1'},
                'b': {'s2'},
            }
        },
        start_state='s0',
        accept_states={'s1'}
    )

    N7 = NFA(
        # Accepts (ab)*
        states={'0', '1', '2', '3'},
        alphabet={'a', 'b'},
        transition_function={
            '0': {
                'a': {'1'},
                'b': {'3'},
            },
            '1': {
                'a': {'3'},
                'b': {'2'},
            },
            '2': {
                'a': {'1'},
                'b': {'3'},
            },
            '3': {
                'a': {'3'},
                'b': {'3'}
            }
        },
        start_state='0',
        accept_states={'2'}
    )

    N8 = NFA(
        # Accepts a*b*
        states={'0', '1', '2'},
        alphabet={'a', 'b'},
        transition_function={
            '0': {
                'a': {'0'},
                'b': {'1'},
            },
            '1': {
                'a': {'2'},
                'b': {'1'},
            },
            '2': {
                'a': {'2'},
                'b': {'2'}
            }
        },
        start_state='0',
        accept_states={'0', '1'}
    )

    N9 = NFA(
        # Accepts a|b
        states={'0', '1', '2', '3', '4'},
        alphabet={'a', 'b'},
        transition_function={
            '0': {
                EMPTY_STRING: {'1', '2'}
            },
            '1': {
                'a': {'3'},
            },
            '2': {
                'b': {'4'}
            }
        },
        start_state='0',
        accept_states={'3', '4'}
    )

    N10 = NFA(
        # DFA of N9
        states={'0', '1', '2', '3'},
        alphabet={'a', 'b'},
        transition_function={
            '0': {
                'a': {'1'},
                'b': {'2'},
            },
            '1': {
                'a': {'3'},
                'b': {'3'}
            },
            '2': {
                'a': {'3'},
                'b': {'3'}
            },
            '3': {
                'a': {'3'},
                'b': {'3'}
            }
        },
        start_state='0',
        accept_states={'1', '2'}
    )

    N11 = NFA(
        # Accepts b*ab*a(a|bb*a)*bb*
        states={'0', '1', '2', '3'},
        alphabet={'a', 'b'},
        transition_function={
            '0': {
                'a': {'0', '1'},
                'b': {'0'},
            },
            '1': {
                'a': {'1', '2'},
                'b': {'1'},
            },
            '2': {
                EMPTY_STRING: {'0'},
                'a': {'2'},
                'b': {'2', '3'},
            },
        },
        start_state='0',
        accept_states={'3'}
    )

    N12 = NFA(
        # DFA of N11
        states={'0', '1', '2', '3'},
        alphabet={'a', 'b'},
        transition_function={
            '0': {
                'a': {'1'},
                'b': {'0'},
            },
            '1': {
                'a': {'2'},
                'b': {'1'},
            },
            '2': {
                'a': {'2'},
                'b': {'3'},
            },
            '3': {
                'a': {'2'},
                'b': {'3'},
            }
        },
        start_state='0',
        accept_states={'3'}
    )

    N13 = NFA(
        # Accept a single 'a'
        states={'0', '1', '2'},
        alphabet={'a', 'b'},
        transition_function={
            '0': {
                'a': {'1'},
                'b': {'2'},
            },
            '1': {
                'a': {'2'},
                'b': {'2'},
            },
            '2': {
                'a': {'2'},
                'b': {'2'},
            },
        },
        start_state='0',
        accept_states={'1'}
    )

    N14 = NFA(
        states={'a', 'b', 'c', 'd', 'e', 'f'},
        alphabet={'0', '1'},
        transition_function={
            'a': {
                '0': {'b'},
                '1': {'c'},
            },
            'b': {
                '0': {'a'},
                '1': {'d'},
            },
            'c': {
                '0': {'e'},
                '1': {'f'},
            },
            'd': {
                '0': {'e'},
                '1': {'f'},
            },
            'f': {
                '0': {'f'},
                '1': {'f'},
            },
        },
        start_state='a',
        accept_states={'c', 'e', 'e'},
    )
