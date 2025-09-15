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
                if symbol != EMPTY_STRING_TRANSITION:
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
        names: dict[States, State] = dict()

        def create_name(state: States) -> State:
            return names.setdefault(
                state,
                '{' + ', '.join(state) + '}'
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
        is_deterministic = True
        for q in self.states:
            q_map = self.transition_function.get(q, dict())
            empty_string_transition_set = q_map.get(
                EMPTY_STRING_TRANSITION, set()
            )
            if not q_map or empty_string_transition_set:
                is_deterministic = False
                break
            else:
                for c in self.alphabet:
                    c_set = q_map.get(c, set())
                    if len(c_set) == 0 or len(c_set) > 1:
                        is_deterministic = False
                        break
                if not is_deterministic:
                    break

        if is_deterministic:
            self_nfa = self
        else:
            self_nfa = self.compute_equivalent_DFA()

        partitions = set()
        X = frozenset(self_nfa.accept_states)
        if X:
            partitions.add(frozenset(X))
        Y = frozenset(self_nfa.states - self_nfa.accept_states)
        if Y:
            partitions.add(frozenset(Y))
        queue = set(partitions)
        while queue:
            A = queue.pop()
            for c in self_nfa.alphabet:
                X = set()
                for q in self_nfa.states:
                    S = self_nfa.move_set({q}, c)
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
        alphabet = set(self_nfa.alphabet)
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
            if P.intersection(self_nfa.accept_states):
                accept_states.add(P_name)
            transition_function[P_name] = {
                c: {states_partitions_map[self_nfa.move_set(P, c).pop()]}
                for c in alphabet
            }
        start_state = states_partitions_map[self_nfa.start_state]
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
        start_state_set: States =\
            transition_function[start_state][EMPTY_STRING_TRANSITION]
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
