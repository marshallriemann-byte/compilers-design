# Regular expressions

from nfa import NFA, Symbol
from enum import Enum
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import override, Self
from copy import deepcopy


class TokenType(Enum):
    EMPTY_STRING_TOKEN = 0  # Empty string
    SYMBOL = 1  # Alphabet symbol
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
class RegularExpressionAST(ABC):
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


class UnionAST(RegularExpressionAST):
    def __init__(self, alternatives: Sequence[RegularExpressionAST]):
        self.alternatives: list[RegularExpressionAST] = list(alternatives)

    def __eq__(self, other):
        match other:
            case UnionAST(alternatives=alts):
                return self.alternatives == alts
            case _:
                return False

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


class ConcatenationAST(RegularExpressionAST):
    def __init__(self, sequence: Sequence[RegularExpressionAST]):
        self.sequence: list[RegularExpressionAST] = list(sequence)

    def __eq__(self, other):
        match other:
            case ConcatenationAST(sequence=seq):
                return self.sequence == seq
            case _:
                return False

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
            f'({str(re)})' if type(re) is UnionAST else str(re)
            for re in self.sequence
        ])


# Quantifiers: ? * + {m} {n,m} {,m} {m,}
class Quantifier(ABC):
    @abstractmethod
    def apply(nfa: NFA) -> NFA:
        pass


class QuantifierStar(Quantifier):
    @override
    def apply(nfa: NFA) -> NFA:
        return NFA.kleene_star(nfa)


class QuantifiedAST(RegularExpressionAST):
    def __init__(self, inner_expr: RegularExpressionAST):
        self.inner_expr: RegularExpressionAST = inner_expr

    def __eq__(self, other):
        match other:
            case QuantifiedAST(inner_expr=expr):
                return self.inner_expr == expr
            case _:
                return False

    @override
    def to_NFA(self) -> NFA:
        return NFA.kleene_star(self.inner_expr.to_NFA())

    @override
    def __repr__(self) -> str:
        return f'Quantified({repr(self.inner_expr)})'

    @override
    def __str__(self) -> str:
        match self.inner_expr:
            case UnionAST() | ConcatenationAST():
                return f'({str(self.inner_expr)})*'
            case _:
                return f'{str(self.inner_expr)}*'


class EmptyStringAST(RegularExpressionAST):
    @override
    def to_NFA(self) -> NFA:
        return NFA(
            states={'q0'},
            alphabet=set(),
            transition_function=dict(),
            start_state='q0',
            accept_states={'q0'},
        )

    def __eq__(self, other):
        return type(other) is EmptyStringAST

    @override
    def __repr__(self) -> str:
        return 'EmptyStringAST()'

    @override
    def __str__(self) -> str:
        return EMPTY_STRING_CHAR


class SymbolAST(RegularExpressionAST):
    def __init__(self, value: Symbol):
        self.value: Symbol = value

    def __eq__(self, other):
        match other:
            case SymbolAST(value=char):
                return self.value == char
            case _:
                return False

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


class GroupAST(RegularExpressionAST):
    def __init__(self, inner_expr: RegularExpressionAST):
        self.inner_expr: RegularExpressionAST = inner_expr

    def __eq__(self, other):
        match other:
            case GroupAST(inner_expr=expr):
                return self.inner_expr == expr
            case _:
                return False

    @override
    def to_NFA(self) -> NFA:
        return self.inner_expr.to_NFA()

    @override
    def __repr__(self) -> str:
        return f'Group({repr(self.inner_expr)})'

    @override
    def __str__(self) -> str:
        return f'({str(self.inner_expr)})'


class EmptyLanguageAST(RegularExpressionAST):
    def __eq__(self, other):
        return type(other) is EmptyLanguageAST

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
        return 'EmptyLanguageAST()'

    @override
    def __str__(self) -> str:
        return 'Φ'


def union(a: RegularExpressionAST, b: RegularExpressionAST) -> RegularExpressionAST:
    match (a, b):
        case (EmptyLanguageAST(), other):
            # ∅ U R = R
            return other
        case (other, EmptyLanguageAST()):
            # R U ∅ = R
            return other
        case (UnionAST(alternatives=alts1), UnionAST(alternatives=alts2)):
            return UnionAST(alternatives=[*alts1, *alts2])
        case (UnionAST(alternatives=alts1), other):
            return UnionAST(alternatives=[*alts1, other])
        case (other, UnionAST(alternatives=alts1)):
            return UnionAST(alternatives=[other, *alts1])
        case (x, y):
            return UnionAST(alternatives=[x, y])


def concatenate(a: RegularExpressionAST, b: RegularExpressionAST) -> RegularExpressionAST:
    match (a, b):
        case (EmptyLanguageAST(), _) | (_, EmptyLanguageAST()):
            # R ∅ = ∅ R = ∅
            return EmptyLanguageAST()
        case (EmptyStringAST(), x) | (x, EmptyStringAST()):
            # R ε = ε R = R
            return x
        case (ConcatenationAST(sequence=seq1), ConcatenationAST(sequence=seq2)):
            return ConcatenationAST(sequence=[*seq1, *seq2])
        case (ConcatenationAST(sequence=seq1), other):
            return ConcatenationAST(sequence=[*seq1, other])
        case (other, ConcatenationAST(sequence=seq1)):
            return ConcatenationAST(sequence=[other, *seq1])
        case (x, y):
            return ConcatenationAST(sequence=[x, y])


def kleene_star(x: RegularExpressionAST) -> RegularExpressionAST:
    match x:
        case EmptyLanguageAST() | EmptyStringAST():
            # ∅* = ε, ε* = ε
            return EmptyStringAST()
        case QuantifiedAST():
            # x = R* => x* = (R*)* = R*
            return x
        case expr:
            return QuantifiedAST(expr)


# Regular expressions context free grammar
# Expression => Concatenation ( '|' Concatenation )*
# Concatenation => Quantified Quantified*
# Quantified => Primary ( '*' )*
# Primary => ε | SYMBOL | Group
# Group => ( '(' Expression ')' )


# Special character to represent empty string
EMPTY_STRING_CHAR = '#'
META_CHARACTERS = {EMPTY_STRING_CHAR, '*', '|', '(', ')', '\\'}


class ParseResult:
    def __init__(self, parsed_expression: RegularExpressionAST, error: str):
        self.parsed_expression: RegularExpressionAST = parsed_expression
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
                        token_type=TokenType.SYMBOL,
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
                        token_type=TokenType.SYMBOL,
                    )
        self.current.pos = begin
        self.pos += 1

    def parse(self) -> RegularExpressionAST:
        if self.current:
            result = self.parse_expression()
            error, parsed_expression = result.error, result.parsed_expression
            if error:
                # Syntax error
                raise ValueError(error)
            elif parsed_expression:
                if error:
                    raise Exception(
                        "\nError & parsed expression\n" +
                        f"error: {error}\n" +
                        f"parsed expression: {parsed_expression}\n"
                    )
                elif self.current:
                    error = f'\nError in position {self.current.pos}\n'
                    error += 'Unexpected item\n'
                    error += f'{self.pattern}\n'
                    error += ' ' * self.current.pos + '^'
                    raise ValueError(error)
                else:
                    # parsing successful, no erorrs
                    return parsed_expression
            elif self.current:
                error = f'\nError in position {self.current.pos}\n'
                error += 'Unexpected item\n'
                error += f'{self.pattern}\n'
                error += ' ' * self.current.pos + '^'
                raise ValueError(error)
        # Empty string pattern
        return EmptyStringAST()

    def check_current_type(self, expected_type: TokenType) -> bool:
        return self.current and self.current.token_type == expected_type

    # Expression => Concatenation ( '|' Concatenation )*
    def parse_expression(self) -> ParseResult:
        initial = self.parse_concatenation()
        error, parsed_expression = initial.error, initial.parsed_expression
        if parsed_expression:
            alternatives: list[RegularExpressionAST] = [parsed_expression]
            while not error and self.check_current_type(TokenType.UNION_BAR):
                self.generate_next_token()  # skip current |
                right_term = self.parse_concatenation()
                if right_term.error:
                    parsed_expression = None
                    error = right_term.error
                elif right_term.parsed_expression:
                    match right_term.parsed_expression:
                        case GroupAST(inner_expr=UnionAST(alternatives=alts)):
                            # Flatten unions
                            alternatives.extend(alts)
                        case other if other not in alternatives:
                            alternatives.append(other)
                else:
                    # Expected expression after |
                    parsed_expression = None
                    error = f'\nError in position {self.current.pos}\n'
                    error += 'Expected expression after |\n'
                    error += f'{self.pattern}\n'
                    error += ' ' * self.current.pos + '^'
            if not error:
                if len(alternatives) == 1:
                    parsed_expression = alternatives.pop()
                else:
                    parsed_expression = UnionAST(alternatives)
            else:
                parsed_expression = None
        return ParseResult(parsed_expression, error)

    # Concatenation => Quantified Quantified*
    def parse_concatenation(self) -> ParseResult:
        initial = self.parse_star()
        error, parsed_expression = initial.error, initial.parsed_expression
        if parsed_expression:
            sequence: list[RegularExpressionAST] = [parsed_expression]
            while not error:
                right_term = self.parse_star()
                if right_term.error:
                    parsed_expression = None
                    error = right_term.error
                elif right_term.parsed_expression:
                    match right_term.parsed_expression:
                        case EmptyStringAST():
                            # Concatenating empty string has no effect
                            pass
                        case GroupAST(
                            inner_expr=ConcatenationAST(sequence=seq)
                        ):
                            # Flatten concatenations
                            sequence.extend(seq)
                        case other:
                            sequence.append(other)
                else:
                    # No more expressions to concatenate
                    break
            if not error:
                if len(sequence) == 1:
                    parsed_expression = sequence.pop()
                else:
                    parsed_expression = ConcatenationAST(sequence)
            else:
                parsed_expression = None
        return ParseResult(parsed_expression, error)

    # Quantified => Primary ( '*' )*
    def parse_star(self) -> ParseResult:
        primary = self.parse_primary()
        if primary.parsed_expression:
            error = None
            parsed_expression = primary.parsed_expression
            while self.check_current_type(TokenType.KLEENE_STAR):
                self.generate_next_token()  # Consume *
                # Avoid stupid parse trees like a*********
                # and (((((epsilon)*)*)*)*)*
                parsed_expression = kleene_star(parsed_expression)
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
                    parsed_expression=EmptyStringAST(),
                    error=None,
                )
            case TokenType.SYMBOL:
                symbol_value = self.current.value
                self.generate_next_token()  # Consume (
                return ParseResult(
                    parsed_expression=SymbolAST(
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
                match parsed_expression:
                    case UnionAST() | ConcatenationAST():
                        # Group only unions and concatenations
                        parsed_expression = GroupAST(
                            inner_expr=parsed_expression
                        )
            else:
                # Expected ) after expression
                parsed_expression = None
                error = f'\nError in position {self.current.pos}\n'
                error += 'Expected ) after expression\n'
                error += f'{self.pattern}\n'
                error += ' ' * self.current.pos + '^'
        else:
            # Expected expression after (
            parsed_expression = None
            error = f'\nError in position {self.current.pos}\n'
            error += 'Expected expression after (\n'
            error += f'{self.pattern}\n'
            error += ' ' * self.current.pos + '^'
        return ParseResult(parsed_expression, error)


class RegularExpression:
    def __init__(self, pattern: str = None, optimize_automaton=False):
        self.pattern: str = pattern

        if self.pattern:
            self.ast: RegularExpressionAST = (
                RegularExpressionParser(pattern).parse()
            )
        else:
            self.ast: RegularExpressionAST = None

        self.is_optimized = False
        if self.pattern:
            self.nfa: NFA = self.ast.to_NFA()
            self.optimize_automaton()
        else:
            self.nfa: NFA = None

    def optimize_automaton(self):
        if not self.is_optimized:
            if not self.nfa.is_minimized:
                self.nfa = (
                    self.nfa.compute_minimized_DFA().rename_states()
                )
            self.is_optimized = True

    def __repr__(self) -> str:
        pattern = f"pattern='{self.pattern}'"
        optimize = f"optimize_automaton='{str(self.is_optimized)}'"
        return f'RegularExpression({pattern}, {optimize})'

    def __str__(self) -> str:
        pattern = f"pattern='{self.pattern}'"
        ast = f"ast='{str(self.ast)}'"
        return f'RE({pattern}, {ast})'

    @staticmethod
    def from_AST(ast: RegularExpressionAST) -> Self:
        re = RegularExpression(None)
        re.pattern = str(ast)
        re.ast = deepcopy(ast)
        re.nfa = re.ast.to_NFA()
        return re


re = RegularExpression(pattern='(a|b)*cbc', optimize_automaton=True)
