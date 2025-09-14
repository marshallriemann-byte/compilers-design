from nfa import NFA, State, EMPTY_STRING_TRANSITION
from regular_expression import RegularExpression
from regular_expression import RegularExpressionParser
from regular_expression import EmptyStringExpression
from regular_expression import EmptyLanguage
from regular_expression import SymbolExpression
from regular_expression import UnionExpression
from regular_expression import Star
from regular_expression import Concatenation
from regular_expression import Group
from copy import deepcopy
from uuid import uuid4
from collections.abc import Sequence
from itertools import permutations


def to_regular_expression_object(pattern: str) -> RegularExpression:
    return RegularExpressionParser(pattern).parse()


def regular_expression_to_NFA(pattern: str) -> NFA:
    return RegularExpressionParser(pattern).parse().to_NFA()


def NFA_to_regular_expression(
        nfa: NFA,
        removal_sequence: Sequence[State] = None
) -> RegularExpression:
    if not removal_sequence:
        removal_sequence = list(nfa.states)

    if nfa.states != set(removal_sequence):
        raise ValueError(
            f'Incomplete removal sequence, missing {
                nfa.states - set(removal_sequence)}'
        )

    # GNFA construction
    table: dict[State, dict[State, RegularExpression]] = dict()
    for (q, q_map) in nfa.transition_function.items():
        q_table = table.setdefault(q, dict())
        for (c, c_set) in q_map.items():
            if c == EMPTY_STRING_TRANSITION:
                c_expr = EmptyStringExpression()
            else:
                c_expr = SymbolExpression(value=c)
            for r in c_set:
                q_table[r] = q_table.get(r, EmptyLanguage()) | c_expr

    gnfa_start_state = f'(GNFA-START, {uuid4().hex})'
    table[gnfa_start_state] = {
        nfa.start_state: EmptyStringExpression()
    }

    gnfa_accept_state = f'(GNFA-ACCEPT, {uuid4().hex})'
    for q in nfa.accept_states:
        q_map = table.setdefault(q, dict())
        q_map[gnfa_accept_state] = EmptyStringExpression()

    for leaving in removal_sequence:
        leaving_map = table.get(leaving, None)
        if not leaving_map:
            continue
        iteration_set = {q for q in nfa.states if q != leaving}
        senders = iteration_set.union({gnfa_start_state})
        for sender in senders:
            sender_map = table.get(sender, None)
            if not sender_map or not sender_map.get(leaving, None):
                continue
            sender_to_leaving = sender_map[leaving]
            receivers = {
                q for q in iteration_set.union({gnfa_accept_state})
                if q != leaving
            }
            for receiver in receivers:
                leaving_to_receiver = leaving_map.get(receiver, None)
                if not leaving_to_receiver:
                    continue
                leaving_self_loop = leaving_map.get(leaving, EmptyLanguage())
                loop = ~leaving_self_loop
                sender_to_receiver = sender_map.get(receiver, EmptyLanguage())
                sender_map[receiver] = sender_to_receiver | (
                    sender_to_leaving +
                    loop +
                    leaving_to_receiver
                )
        del table[leaving]
        for (_, q_map) in table.items():
            try:
                del q_map[leaving]
            except KeyError:
                pass
    re = table[gnfa_start_state].get(gnfa_accept_state, EmptyLanguage())
    if isinstance(re, EmptyLanguage):
        print('This NFA has empty language')
    return re


def get_all_regular_expressions(nfa: NFA) -> list[RegularExpression]:
    return [
        NFA_to_regular_expression(nfa, states_permutation)
        for states_permutation in permutations(list(nfa.states))
    ]


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
            EMPTY_STRING_TRANSITION: {'1', '7'},
        },
        '1': {
            EMPTY_STRING_TRANSITION: {'2', '4'},
        },
        '2': {
            'a': {'3'},
        },
        '3': {
            EMPTY_STRING_TRANSITION: {'6'},
        },
        '4': {
            'b': {'5'},
        },
        '5': {
            EMPTY_STRING_TRANSITION: {'6'},
        },
        '6': {
            EMPTY_STRING_TRANSITION: {'1', '7'},
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
            EMPTY_STRING_TRANSITION: {'1', '2'}
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
            EMPTY_STRING_TRANSITION: {'0'},
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

N15 = NFA(
    states={'0', '1', '2', '3', '4', '5', '6'},
    alphabet={'a', 'b'},
    transition_function={
        '0': {
            'a': {'1'},
            'b': {'3'},
        },
        '1': {
            'a': {'2'},
            'b': {'3'},
        },
        '2': {
            'a': {'3'},
            'b': {'4'},
        },
        '3': {
            'a': {'1'},
            'b': {'4'},
        },
        '4': {
            'a': {'1'},
            'b': {'5'},
        },
        '5': {
            'a': {'1'},
            'b': {'5'},
        },
        '6': {
            'a': {'6'},
            'b': {'3'},
        },
    },
    start_state='0',
    accept_states={'2', '4'}
)

used = N15.compute_minimized_DFA()
print(vars(used))
