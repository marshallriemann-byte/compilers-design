import main
import nfa
import regular_expression

N1 = nfa.NFA(
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

N2 = nfa.NFA(
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

N3 = nfa.NFA(
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

N4 = nfa.NFA(
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

N5 = nfa.NFA(
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

N6 = nfa.NFA(
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

N7 = nfa.NFA(
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

N8 = nfa.NFA(
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

N9 = nfa.NFA(
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

N10 = nfa.NFA(
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

N11 = nfa.NFA(
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

N12 = nfa.NFA(
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

N13 = nfa.NFA(
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

N14 = nfa.NFA(
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

N15 = nfa.NFA(
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
