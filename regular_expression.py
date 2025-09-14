# Regular expressions

from nfa import NFA, Symbol
from enum import Enum
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import override, Self


class TokenType(Enum):
    EMPTY_STRING_TOKEN = 0  # Empty string
    SYBMOL = 1  # Alphabet symbol
    LEFT_PARENTHESIS = 3  # (
    RIGHT_PARENTHESIS = 4  # )
    UNION_BAR = 5  # |
    KLEENE_STAR = 6  # *


class Token:
    def __init__(self, value: str, token_type: TokenType, pos: int = 0):
        self.value: str = value
        self.token_type: TokenType = token_type
        self.pos: int = pos

    def __repr__(self):
        value = f"Token(value='{self.value}', "
        value += f'type={self.token_type}, '
        value += f'pos={self.pos})'
        return value


# Regular expressions abtract base class
class RegularExpression(ABC):
    @abstractmethod
    def to_NFA(self) -> NFA:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    def __add__(self, other: Self) -> Self:
        return concatenate(self, other)

    def __or__(self, other: Self) -> Self:
        return union(self, other)

    def __invert__(self) -> Self:
        return kleene_star(self)


class UnionExpression(RegularExpression):
    def __init__(self, alternatives: Sequence[RegularExpression]):
        self.alternatives: list[RegularExpression] = list(alternatives)

    @override
    def to_NFA(self) -> NFA:
        return NFA.union([
            re.to_NFA()
            for re in self.alternatives
        ])

    @override
    def __repr__(self) -> str:
        value = ', '.join([repr(re) for re in self.alternatives])
        return f'UnionExpression({value})'

    @override
    def __str__(self) -> str:
        return '|'.join([str(re) for re in self.alternatives])


class Concatenation(RegularExpression):
    def __init__(self, sequence: Sequence[RegularExpression]):
        self.sequence: list[RegularExpression] = list(sequence)

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
        return ''.join([
            f'({str(re)})' if isinstance(re, UnionExpression) else str(re)
            for re in self.sequence
        ])


class Star(RegularExpression):
    def __init__(self, expr: RegularExpression):
        self.expr: RegularExpression = expr

    @override
    def to_NFA(self) -> NFA:
        return NFA.kleene_star(self.expr.to_NFA())

    @override
    def __repr__(self) -> str:
        return f'Star({repr(self.expr)})'

    @override
    def __str__(self) -> str:
        match self.expr:
            case UnionExpression() | Concatenation():
                return f'({str(self.expr)})*'
            case _:
                return f'{str(self.expr)}*'


class EmptyStringExpression(RegularExpression):
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
        return 'EmptyStringExpression()'

    @override
    def __str__(self) -> str:
        return EMPTY_STRING_CHAR


class SymbolExpression(RegularExpression):
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


class Group(RegularExpression):
    def __init__(self, grouped_expr: RegularExpression):
        self.grouped_expr: RegularExpression = grouped_expr

    @override
    def to_NFA(self) -> NFA:
        return self.grouped_expr.to_NFA()

    @override
    def __repr__(self) -> str:
        return f'Group({repr(self.grouped_expr)})'

    @override
    def __str__(self) -> str:
        return f'({str(self.grouped_expr)})'


class EmptyLanguage(RegularExpression):
    @override
    def to_NFA(self) -> NFA:
        return NFA(
            states={'q0'},
            alphabet=set(),
            transition_function=dict(),
            start_state='q0',
            accept_states=set()
        )

    @override
    def __repr__(self) -> str:
        return 'EmptyLanguage()'

    @override
    def __str__(self) -> str:
        return 'Φ'


def union(a: RegularExpression, b: RegularExpression) -> RegularExpression:
    match (a, b):
        case (EmptyLanguage(), other):
            # ∅ U R = R
            return other
        case (other, EmptyLanguage()):
            # R U ∅ = R
            return other
        case (UnionExpression(alternatives=alts1), UnionExpression(alternatives=alts2)):
            return UnionExpression(alternatives=[*alts1, *alts2])
        case (UnionExpression(alternatives=alts1), other):
            return UnionExpression(alternatives=[*alts1, other])
        case (other, UnionExpression(alternatives=alts1)):
            return UnionExpression(alternatives=[other, *alts1])
        case (x, y):
            return UnionExpression(alternatives=[x, y])


def concatenate(a: RegularExpression, b: RegularExpression) -> RegularExpression:
    match (a, b):
        case (EmptyLanguage(), _) | (_, EmptyLanguage()):
            # R ∅ = ∅ R = ∅
            return EmptyLanguage()
        case (EmptyStringExpression(), x) | (x, EmptyStringExpression()):
            # R ε = ε R = R
            return x
        case (Concatenation(sequence=seq1), Concatenation(sequence=seq2)):
            return Concatenation(sequence=[*seq1, *seq2])
        case (Concatenation(sequence=seq1), other):
            return Concatenation(sequence=[*seq1, other])
        case (other, Concatenation(sequence=seq1)):
            return Concatenation(sequence=[other, *seq1])
        case (x, y):
            return Concatenation(sequence=[x, y])


def kleene_star(x: RegularExpression) -> RegularExpression:
    match x:
        case EmptyLanguage() | EmptyStringExpression():
            # ∅* = ε, ε* = ε
            return EmptyStringExpression()
        case Star(expr=_):
            # x = R* => x* = (R*)* = R*
            return x
        case expr:
            return Star(expr)


# Regular expressions context free grammar
# Expression => Concatenation ( '|' Concatenation )*
# Concatenation => Star Star*
# Star => Primary ( '*' )*
# Primary => ε | SYMBOL | Group
# Group => ( '(' Expression ')' )


# Special character to represent empty string
EMPTY_STRING_CHAR = '#'
META_CHARACTERS = {EMPTY_STRING_CHAR, '*', '|', '(', ')', '\\'}


class ParseResult:
    def __init__(self, parsed_expression: RegularExpression, error: str):
        self.parsed_expression: RegularExpression = parsed_expression
        self.error: str = error


class RegularExpressionParser:
    def __init__(self, pattern: str):
        self.pattern: str = pattern
        self.pos = 0
        self.current: Token = None
        self.generate_next_token()

    def generate_next_token(self):
        if self.pos >= len(self.pattern):
            self.current = None
            return
        begin = self.pos
        match self.pattern[self.pos]:
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
                    value=')',
                    token_type=TokenType.RIGHT_PARENTHESIS,
                )
            case '\\':
                if self.pos+1 < len(self.pattern):
                    char = self.pattern[self.pos+1]
                    if char in META_CHARACTERS:
                        # Metacharacter symbol
                        self.pos += 1  # Skip escape backslash
                    else:
                        # ignore next char, emit backslash as symbol
                        char = '\\'
                    self.current = Token(
                        value=char,
                        token_type=TokenType.SYBMOL,
                    )
                else:
                    raise ValueError('Trailing slash at pattern end')
            case c:
                if c == EMPTY_STRING_CHAR:
                    self.current = Token(
                        value=EMPTY_STRING_CHAR,
                        token_type=TokenType.EMPTY_STRING_TOKEN,
                    )
                else:   # any other character
                    self.current = Token(
                        value=c,
                        token_type=TokenType.SYBMOL,
                    )
        self.current.pos = begin
        self.pos += 1

    def parse(self) -> RegularExpression:
        if self.current:
            result = self.parse_expression()
            error, parsed_expression = result.error, result.parsed_expression
            if parsed_expression:
                if self.current:
                    error = f'Error in position {self.pos}\n'
                    error += 'Unexpected item\n'
                    error += f'{self.pattern}\n'
                    error += ' ' * self.current.pos + '^'
                    raise ValueError(error)
                elif error:
                    raise Exception(
                        "Error & parsed expression\n" +
                        f"error: {error}\n" +
                        f"parsed expression: {parsed_expression}\n"
                    )
                else:
                    # parsing successful, no erorrs
                    return parsed_expression
            elif error:
                raise ValueError(error)
        # Empty string pattern
        return EmptyStringExpression()

    def check_current_type(self, expected_type: TokenType) -> bool:
        return self.current and self.current.token_type == expected_type

    # Expression => Concatenation ( '|' Concatenation )*
    def parse_expression(self) -> ParseResult:
        initial = self.parse_concatenation()
        error, parsed_expression = initial.error, initial.parsed_expression
        if parsed_expression:
            alternatives: list[RegularExpression] = [parsed_expression]
            while not error and self.check_current_type(TokenType.UNION_BAR):
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
                    error += ' ' * self.pos + '^'
            if not error:
                if len(alternatives) == 1:
                    parsed_expression = alternatives.pop()
                else:
                    parsed_expression = UnionExpression(alternatives)
            else:
                parsed_expression = None
        return ParseResult(parsed_expression, error)

    # Concatenation => Star Star*
    def parse_concatenation(self) -> ParseResult:
        initial = self.parse_star()
        error, parsed_expression = initial.error, initial.parsed_expression
        if parsed_expression:
            sequence: list[RegularExpression] = [parsed_expression]
            while not error:
                right_term = self.parse_star()
                if right_term.error:
                    parsed_expression = None
                    error = right_term.error
                elif right_term.parsed_expression:
                    sequence.append(right_term.parsed_expression)
                else:
                    # No more expressions to concatenate
                    break
            if not error:
                if len(sequence) == 1:
                    parsed_expression = sequence.pop()
                else:
                    parsed_expression = Concatenation(sequence)
            else:
                parsed_expression = None
        return ParseResult(parsed_expression, error)

    # Star => Primary ( '*' )*
    def parse_star(self) -> ParseResult:
        primary = self.parse_primary()
        if primary.parsed_expression:
            error = None
            parsed_expression = primary.parsed_expression
            while self.check_current_type(TokenType.KLEENE_STAR):
                self.generate_next_token()  # Consume *
                parsed_expression = Star(
                    expr=parsed_expression
                )
        else:
            parsed_expression = None
            error = primary.error
        return ParseResult(parsed_expression, error)

    # Primary => ε | SYMBOL | ( '(' Expression ')' )
    def parse_primary(self) -> ParseResult:
        current_type = None if not self.current else self.current.token_type
        match current_type:
            case TokenType.EMPTY_STRING_TOKEN:
                self.generate_next_token()
                return ParseResult(
                    parsed_expression=EmptyStringExpression(),
                    error=None,
                )
            case TokenType.SYBMOL:
                symbol_value = self.current.value
                self.generate_next_token()  # Consume (
                return ParseResult(
                    parsed_expression=SymbolExpression(
                        value=symbol_value
                    ),
                    error=None
                )
            case TokenType.LEFT_PARENTHESIS:
                self.generate_next_token()  # Consume (
                return self.parse_group()
            case _:
                # No token avaliable, or a weird token
                return ParseResult(
                    parsed_expression=None,
                    error=None
                )

    # Group => ( '(' Expression ')' )
    def parse_group(self) -> ParseResult:
        expr = self.parse_expression()
        error, parsed_expression = expr.error, expr.parsed_expression
        if error:
            parsed_expression = None
        elif parsed_expression:
            if self.check_current_type(TokenType.RIGHT_PARENTHESIS):
                self.generate_next_token()  # Consume )
                parsed_expression = Group(
                    grouped_expr=parsed_expression
                )
            else:
                # Expected ) after expression
                parsed_expression = None
                error = f'Error in position {self.pos}\n'
                error += 'Expected ) after expression\n'
                error += f'{self.pattern}\n'
                error += ' ' * self.pos + '^'
        else:
            # Expected expression after (
            parsed_expression = None
            error = f'Error in position {self.pos}\n'
            error += 'Expected expression after (\n'
            error += f'{self.pattern}\n'
            error += ' ' * self.pos + '^'
        return ParseResult(parsed_expression, error)
