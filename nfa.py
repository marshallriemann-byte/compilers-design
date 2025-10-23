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


EMPTY_STRING_TRANSITION = ''


class AutomatonType(Enum):
    NONDETERMINISTIC = 0
    DETERMINISTIC = 1


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
        is_deterministic = True
        for (q, q_map) in self.transition_function.items():
            self.states.add(q)
            for (symbol, symbol_set) in q_map.items():
                if symbol != EMPTY_STRING_TRANSITION:
                    self.alphabet.add(symbol)
                    if len(symbol_set) > 1:
                        # This symbol leads to multiple states
                        is_deterministic = False
                elif symbol_set:
                    # This state has at least one empty string transition
                    is_deterministic = False
                self.states.update(symbol_set)
        for (_, q_map) in self.transition_function.items():
            state_transitions = len(q_map)
            if EMPTY_STRING_TRANSITION in q_map:
                state_transitions -= 1
            if state_transitions < len(self.alphabet):
                # This state does not have transitions for all symbols
                is_deterministic = False
            if not is_deterministic:
                break
        self.start_state = start_state
        self.states.add(self.start_state)
        self.accept_states = set(accept_states)
        self.states.update(self.accept_states)
        if len(self.transition_function) < len(self.states):
            # This NFA does not have transitions for all states
            is_deterministic = False
        if is_deterministic:
            self.automaton_type = AutomatonType.DETERMINISTIC
        else:
            self.automaton_type = AutomatonType.NONDETERMINISTIC
        self.is_minimized = False

    def __repr__(self):
        if self.automaton_type == AutomatonType.DETERMINISTIC:
            label = "DFA"
        else:
            label = "NFA"
        attributes = ', '.join(
            f'{key}={value}'
            for (key, value) in vars(self).items()
        )
        return label + (
            f'({attributes})'
        )

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
            q = queue.popleft()
            for s in self.read_transition(q, EMPTY_STRING_TRANSITION):
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
            s = queue.popleft()
            if self.compute(s) == ComputationResult.ACCEPT:
                print(s)
            for c in self.alphabet:
                queue.append(s + c)

    def compute_equivalent_DFA(self) -> Self:
        if self.automaton_type == AutomatonType.DETERMINISTIC:
            return deepcopy(self)

        names: dict[States, State] = dict()

        def create_name(state: States) -> State:
            return names.setdefault(
                state,
                '{' + ', '.join(sorted(state)) + '}'
            )

        dfa_start_state = frozenset(self.epsilon_closure({self.start_state}))

        dfa_alphabet = {
            s for s in self.alphabet
            if s != EMPTY_STRING_TRANSITION
        }

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
        if self.is_minimized:
            return deepcopy(self)
        elif self.automaton_type == AutomatonType.NONDETERMINISTIC:
            source_nfa = self.compute_equivalent_DFA()
        elif self.automaton_type == AutomatonType.DETERMINISTIC:
            source_nfa = self

        partitions = set()
        X = frozenset(source_nfa.accept_states)
        if X:
            partitions.add(frozenset(X))
        Y = frozenset(source_nfa.states - source_nfa.accept_states)
        if Y:
            partitions.add(frozenset(Y))
        queue = set(partitions)
        while queue:
            A = queue.pop()
            for c in source_nfa.alphabet:
                X = set()
                for q in source_nfa.states:
                    S = source_nfa.move_set({q}, c)
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
                '{' + ', '.join(sorted(state)) + '}'
            )

        states: States = set()
        alphabet = set(source_nfa.alphabet)
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
            if P.intersection(source_nfa.accept_states):
                accept_states.add(P_name)
            transition_function[P_name] = {
                c: {states_partitions_map[source_nfa.move_set(P, c).pop()]}
                for c in alphabet
            }
        start_state = states_partitions_map[source_nfa.start_state]

        minimized_DFA = NFA(
            states,
            alphabet,
            transition_function,
            start_state,
            accept_states
        )
        minimized_DFA.is_minimized = True
        return minimized_DFA

    def rename_states(self) -> Self:
        counter = 0
        names: dict[State, int] = dict()
        for q in sorted(self.states):
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

    # Operators overloading
    def __add__(self, other: Self) -> Self:
        return NFA.concatenate([self, other])

    def __or__(self, other: Self) -> Self:
        return NFA.union([self, other])

    def __invert__(self) -> Self:
        return NFA.kleene_star(self)
    # Operators overloading end

    # Thomspon constructions
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
                EMPTY_STRING_TRANSITION, set()
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
                        EMPTY_STRING_TRANSITION, set()
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
            EMPTY_STRING_TRANSITION: set()
        }
        start_state_set: States = (
            transition_function[start_state][EMPTY_STRING_TRANSITION]
        )
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
    # Thomspon constructions end

    # Special NFAs
    @staticmethod
    def empty_language_NFA() -> Self:
        return NFA(
            states={'q0'},
            alphabet=set(),
            transition_function=dict(),
            start_state='q0',
            accept_states=set()
        )

    @staticmethod
    def empty_string_language_NFA() -> Self:
        return NFA(
            states={'q0'},
            alphabet=set(),
            transition_function=dict(),
            start_state='q0',
            accept_states={'q0'},
        )

    @staticmethod
    def union_empty_string(nfa: Self) -> Self:
        new_start_state = f'(EMPTY, {uuid4().hex})'
        new_transition_function = deepcopy(nfa.transition_function)
        new_transition_function[new_start_state] = {
            EMPTY_STRING_TRANSITION: {nfa.start_state}
        }
        return NFA(
            states=nfa.states.union({new_start_state}),
            alphabet=set(nfa.alphabet),
            transition_function=new_transition_function,
            start_state=new_start_state,
            accept_states=nfa.accept_states.union({new_start_state})
        )
    # Special NFAs end

    # Quantifiers
    # Let L = language of NFA N
    @staticmethod
    def kleene_plus(nfa: Self) -> Self:
        # L+ = L L*
        return NFA.concatenate([nfa, NFA.kleene_star(nfa)])

    @staticmethod
    def power(nfa: Self, exponent: int) -> Self:
        match exponent:
            case 0:
                return NFA.empty_string_language_NFA()
            case 1:
                return nfa
            case m:
                result = nfa
                for _ in range(m-1):
                    result = NFA.concatenate([result, nfa])
                return result

    @staticmethod
    def at_least_NFA(nfa: Self, min_count) -> Self:
        # L{m,} = (L^m) L*
        return NFA.concatenate([
            NFA.power(nfa, min_count), NFA.kleene_star(nfa)
        ])

    @staticmethod
    def at_most_NFA(nfa: Self, max_count) -> Self:
        # L{,n} = L{0,n}
        start_state = f'{nfa.start_state}_0'
        final_state = f'(ACCEPT, {uuid4().hex})'
        states: States = {final_state}
        alphabet = set(nfa.alphabet)
        transition_function: TransitionFunction = dict()
        for i in range(max_count):
            for q, q_map in nfa.transition_function.items():
                q_name = f'{q}_{i}'
                states.add(q_name)
                transition_function[q_name] = dict()
                for c, targets in q_map.items():
                    targets = {f'{t}_{i}' for t in targets}
                    states.update(targets)
                    transition_function[q_name][c] = targets
            for q in nfa.accept_states:
                q_name = f'{q}_{i}'
                transition_function.setdefault(
                    q_name, dict()
                ).setdefault(
                    EMPTY_STRING_TRANSITION, set()
                ).add(final_state)
                if i+1 < max_count:
                    transition_function[q_name][
                        EMPTY_STRING_TRANSITION
                    ].add(f'{nfa.start_state}_{i+1}')
        transition_function.setdefault(
            start_state, dict()
        ).setdefault(
            EMPTY_STRING_TRANSITION, set()
        ).add(final_state)
        accept_states: States = {final_state}
        return NFA(
            states,
            alphabet,
            transition_function,
            start_state,
            accept_states
        )

    @staticmethod
    def bounded_NFA(nfa: Self, min_count, max_count) -> Self:
        # L{m,n}
        return NFA.concatenate([
            NFA.power(nfa, min_count),
            NFA.at_most_NFA(nfa, max_count-min_count)
        ])
    # Quantifiers end

    def __pow__(self, exponent) -> Self:
        # L = language of NFA (self)
        match exponent:
            case int(m):
                # L^m
                return NFA.power(self, m)
            case (int(m), int(n)) if m == n:
                # L^m
                return NFA.power(self, m)
            case (0, 0):
                # L{0, 0} = (L^0) = {empty string}
                return NFA.empty_string_language_NFA()
            case (0, None):
                # L{0,} = L*
                return NFA.kleene_star(self)
            case (0, int(n)):
                # L{,n} = L{0,n}
                return NFA.at_most_NFA(self)
            case (1, None):
                # L+ = L L*
                return NFA.kleene_plus(self)
            case (int(m), None):
                # L{m,} = (L^m) L*
                return NFA.at_least_NFA(self)
            case (int(m), int(n)) if m < n:
                return NFA.bounded_NFA(m, n)
            case _:
                raise TypeError(
                    "Exponent must be an int or a tuple (m, n) with m < n"
                )

    @staticmethod
    def are_equivalent(nfa1: Self, nfa2: Self) -> bool:
        if nfa1.alphabet != nfa2.alphabet:
            return False
        queue = deque([(
            frozenset(nfa1.epsilon_closure({nfa1.start_state})),
            frozenset(nfa2.epsilon_closure({nfa2.start_state}))
        )])
        visited = set()
        while queue:
            s1, s2 = queue.popleft()
            if (s1, s2) in visited:
                continue
            visited.add((s1, s2))
            if (
                bool(s1.intersection(nfa1.accept_states)) ^
                bool(s2.intersection(nfa2.accept_states))
            ):
                return False
            for c in nfa1.alphabet:
                queue.append((
                    frozenset(nfa1.epsilon_closure({nfa1.start_state})),
                    frozenset(nfa2.epsilon_closure({nfa2.start_state})),
                ))
        return True
