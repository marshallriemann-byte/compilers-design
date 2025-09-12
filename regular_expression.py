from nfa import NFA, EMPTY_STRING
from enum import Enum
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import override


class TokenType(Enum):
    EPSILON = 0  # ε Empty string
    SYBMOL = 1  # Alphabet symbol
    BLACKSLASH = 2  # \
    LEFT_PARENTHESIS = 3  # (
    RIGHT_PARENTHESIS = 4  # )
    UNION_BAR = 5  # |
    KLEENE_STAR = 6  # *


class Token:
    def __init__(self, value: str, token_type: TokenType, pos: int):
        self.value: str = value
        self.token_type: TokenType = token_type
        self.pos: int = pos

    def __repr__(self):
        value = f"Token(value='{self.value}', "
        value += f'type={self.token_type}, '
        value += f'pos={self.pos})'
        return value


# Regular expressions abtract base class
class RegeularExpression(ABC):
    @abstractmethod
    def to_NFA(self) -> NFA:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class Union(RegeularExpression):
    def __init__(self, alternatives: Sequence[RegeularExpression]):
        self.alternatives: list[RegeularExpression] = list(alternatives)

    @override
    def to_NFA(self) -> NFA:
        return NFA.union([
            re.to_NFA()
            for re in self.alternatives
        ])

    @override
    def __repr__(self) -> str:
        value = ', '.join([repr(re) for re in self.alternatives])
        return f'Union({value})'

    @override
    def __str__(self) -> str:
        return '|'.join([str(re) for re in self.alternatives])


class Concatenation(RegeularExpression):
    def __init__(self, sequence: Sequence[RegeularExpression]):
        self.sequence: list[RegeularExpression] = list(sequence)

    @override
    def to_NFA(self) -> NFA:
        return NFA.concatenate([
            re.to_NFA()
            for re in self.sequence
        ])

    @override
    def __repr__(self) -> str:
        value = ', '.join([repr(re) for re in self.sequence])
        return f'Concatenation({value})'

    @override
    def __str__(self) -> str:
        return ''.join([str(re) for re in self.sequence])


class Star(RegeularExpression):
    def __init__(self, expr: RegeularExpression):
        self.expr: RegeularExpression = expr

    @override
    def to_NFA(self) -> NFA:
        return NFA.kleene_star(self.expr.to_NFA())

    @override
    def __repr__(self) -> str:
        return f'Star({repr(self.expr)})'

    @override
    def __str__(self) -> str:
        return f'{str(self.expr)}*'


class EmptyString(RegeularExpression):
    @override
    def to_NFA(self) -> NFA:
        return NFA(
            states={'q0'},
            alphabet=set(),
            transition_function=dict(),
            start_state='q0',
            accept_states={'q0'},
        )

    @override
    def __repr__(self) -> str:
        return 'EmptyString()'

    @override
    def __str__(self) -> str:
        return 'ε'
