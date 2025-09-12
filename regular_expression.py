from enum import Enum
from abc import ABC, abstractmethod


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
    def to_NFA(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass
