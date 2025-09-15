from nfa import NFA, State, EMPTY_STRING_TRANSITION
from regular_expression import RegularExpression
from regular_expression import EmptyStringAST
from regular_expression import EmptyLanguageAST
from regular_expression import SymbolAST
from uuid import uuid4
from collections.abc import Sequence
from itertools import permutations


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
                c_expr = EmptyStringAST()
            else:
                c_expr = SymbolAST(value=c)
            for r in c_set:
                q_table[r] = q_table.get(r, EmptyLanguageAST()) | c_expr

    gnfa_start_state = f'(GNFA-START, {uuid4().hex})'
    table[gnfa_start_state] = {
        nfa.start_state: EmptyStringAST()
    }

    gnfa_accept_state = f'(GNFA-ACCEPT, {uuid4().hex})'
    for q in nfa.accept_states:
        q_map = table.setdefault(q, dict())
        q_map[gnfa_accept_state] = EmptyStringAST()

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
                leaving_self_loop = leaving_map.get(
                    leaving, EmptyLanguageAST())
                loop = ~leaving_self_loop
                sender_to_receiver = sender_map.get(
                    receiver, EmptyLanguageAST())
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
    re = table[gnfa_start_state].get(gnfa_accept_state, EmptyLanguageAST())
    if isinstance(re, EmptyLanguageAST):
        print('This NFA has empty language')
    return RegularExpression.from_AST(re)


def get_all_regular_expressions(nfa: NFA) -> list[RegularExpression]:
    return [
        NFA_to_regular_expression(nfa, states_permutation)
        for states_permutation in permutations(list(nfa.states))
    ]
