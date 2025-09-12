from nfa import NFA, EMPTY_STRING, Symbol
from enum import Enum
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import override
from sys import stderr


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


class SymbolExpression(RegeularExpression):
    def __init__(self, value: Symbol):
        self.value: Symbol = value

    @override
    def to_NFA(self) -> NFA:
        return NFA(
            states={'q0', 'q1'},
            alphabet={self.value},
            transition_function={
                'q0': {
                    self.value: {'q1'}
                }
            },
            start_state='q0',
            accept_states={'q1'}
        )

    @override
    def __repr__(self) -> str:
        return f'Symbol({repr(self.value)})'

    @override
    def __str__(self) -> str:
        return self.value


class Group(RegeularExpression):
    def __init__(self, grouped_expr: RegeularExpression):
        self.grouped_expr: RegeularExpression = grouped_expr

    @override
    def to_NFA(self) -> NFA:
        return self.grouped_expr.to_NFA()

    @override
    def __repr__(self) -> str:
        return f'Group({repr(self.grouped_expr)})'

    @override
    def __str__(self) -> str:
        return f'({str(self.grouped_expr)})'


# Regular expressions context free grammar
# Expression => Concatenation ( '|' Concatenation )*
# Concatenation => Star Star*
# Star => Primary ( '*' )?
# Primary => ε | SYMBOL | ( '(' Primary ')' )


META_CHARACTERS = {'*', '|', '(', ')', '\\'}


class ParseResult:
    def __init__(self, parsed_expression: RegeularExpression, error: str):
        self.parsed_expression: RegeularExpression = parsed_expression
        self.error: str = error


class RegularExpressionParser:
    def __init__(self, pattern: str):
        self.pattern: str = pattern
        self.pos = 0
        self.current: Token = None
        self.generate_next_token()

    def generate_next_token(self):
        match self.pattern[self.pos]:
            case 'ε':
                self.current = Token(
                    value='ε',
                    token_type=TokenType.EMPTY_STRING,
                )
            case '*':
                self.current = Token(
                    value='*',
                    token_type=TokenType.KLEENE_STAR,
                )
            case '|':
                self.current = Token(
                    value='|',
                    token_type=TokenType.UNION_BAR,
                )
            case '(':
                self.current = Token(
                    value='(',
                    token_type=TokenType.LEFT_PARENTHESIS,
                )
            case ')':
                self.current = Token(
                    value='(',
                    token_type=TokenType.RIGHT_PARENTHESIS,
                )
            case '\\':
                if self.pos+1 < len(self.pattern):
                    char = self.pattern[self.pos+1]
                    if char in META_CHARACTERS:
                        self.current = Token(
                            value=char,
                            token_type=TokenType.SYBMOL,
                        )
                    else:
                        self.current = Token(
                            value='\\',
                            token_type=TokenType.BLACKSLASH,
                        )
                else:
                    print('Trailing slash at pattern end', file=stderr)
                    exit(1)
            case c:  # any other character
                self.current = Token(
                    value=c,
                    token_type=TokenType.SYBMOL,
                )
        if self.current:
            self.current.pos = self.pos
            self.pos += len(self.current)

    def parse(self) -> RegeularExpression:
        if self.current:
            result = self.parse_expression()
            error, parsed_expression = result.error, result.parsed_expression
            if parsed_expression:
                if error:
                    raise Exception(
                        "Error & parsed expression\n" +
                        f"error: {error}\n" +
                        f"parsed expression: {parsed_expression}\n"
                    )
                else:
                    # parsing successful, no erorrs
                    return parsed_expression
            elif error:
                print(error, file=stderr)
                exit(1)
        # Empty string pattern
        return EmptyString()

    # Expression => Concatenation ( '|' Concatenation )*
    def parse_expression(self) -> ParseResult:
        initial = self.parse_concatenation()
        error, parsed_expression = initial.error, initial.parsed_expression
        if parsed_expression:
            alternatives: list[RegeularExpression] = [parsed_expression]
            while not error and self.current.token_type == TokenType.UNION_BAR:
                self.generate_next_token()  # skip current |
                right_term = self.parse_concatenation()
                if right_term.error:
                    parsed_expression = None
                    error = right_term.error
                elif right_term.parsed_expression:
                    alternatives.append(right_term.parsed_expression)
                else:
                    # Expected expression after |
                    parsed_expression = None
                    error = f'Error in position {self.pos}\n'
                    error += 'Expected expression after |\n'
                    error += f'{self.pattern}\n'
                    error += ' ' * self.pos
            if len(alternatives) == 1:
                parsed_expression = alternatives.pop()
            elif len(alternatives) > 1:
                parsed_expression = Union(alternatives)
        return ParseResult(parsed_expression, error)

    # Concatenation => Star Star*
    def parse_concatenation(self) -> ParseResult:
        initial = self.parse_star()
        error, parsed_expression = initial.error, initial.parsed_expression
        if parsed_expression:
            sequence: list[RegeularExpression] = [parsed_expression]
            while not error or not parsed_expression:
                self.generate_next_token()
                right_term = self.parse_star()
                if right_term.error:
                    parsed_expression = None
                    error = right_term.error
                elif right_term.parsed_expression:
                    sequence.append(right_term.parsed_expression)
            if len(sequence) == 1:
                parsed_expression = sequence.pop()
            elif len(sequence) > 1:
                parsed_expression = Concatenation(sequence)
        return ParseResult(parsed_expression, error)

    # Star => Primary ( '*' )?
    def parse_star(self) -> ParseResult:
        primary = self.parse_primary()
        if primary.error:
            return primary
        self.generate_next_token()
        if self.current.token_type == TokenType.KLEENE_STAR:
            return Star(expr=primary.parsed_expression)
        return primary

    # Primary => ε | SYMBOL | ( '(' Primary ')' )
    def parse_primary(self) -> ParseResult:
        match self.current.token_type:
            case TokenType.EPSILON:
                return EmptyString()
            case TokenType.SYBMOL:
                return SymbolExpression(
                    value=self.current.value
                )
            case TokenType.LEFT_PARENTHESIS:
                return self.parse_group()

    def parse_group(self) -> ParseResult:
        return None
