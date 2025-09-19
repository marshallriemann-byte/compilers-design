# Regular expressions

from nfa import NFA, Symbol
from enum import Enum
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import override, Self
from copy import deepcopy
from dataclasses import dataclass
from re import compile


class TokenType(Enum):
    EMPTY_STRING_TOKEN = 0  # Empty string
    SYMBOL = 1  # Alphabet symbol
    LEFT_PARENTHESIS = 3  # (
    RIGHT_PARENTHESIS = 4  # )
    UNION_BAR = 5  # |
    KLEENE_STAR = 6  # *
    KLEENE_PLUS = 7  # +
    MARK = 8  # ?
    LEFT_CURLY_BRACE = 9  # {
    RIGHT_CURLY_BRACE = 10  # }
    COMMA = 11  # ,
    NUMBER = 12


class BasicToken:
    def __init__(self, token_type: TokenType, pos: int = 0):
        self.token_type: TokenType = token_type
        self.pos: int = pos

    def __repr__(self):
        value = f"BasicToken(type={self.token_type}, "
        value += f'pos={self.pos})'
        return value


class SymbolToken(BasicToken):
    def __int__(self, value: str):
        self.value: str = value

    def __repr__(self):
        value = f"SymbolToken(value='{self.value}', "
        value += f'type={self.token_type}, '
        value += f'pos={self.pos})'
        return value


class NumberToken(BasicToken):
    def __int__(self, value: int):
        self.value: int = value

    def __repr__(self):
        value = f'NumberToken(value={self.value}, '
        value += f'type={self.token_type}, '
        value += f'pos={self.pos})'
        return value


type Token = BasicToken | SymbolToken | NumberToken


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
    def __eq__(self, other) -> bool:
        return type(other) is type(self)

    def __hash__(self, other) -> bool:
        return hash(type(self))

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'

    @abstractmethod
    def apply_on_NFA(self, nfa: NFA) -> NFA:
        pass

    @abstractmethod
    def apply_on_expression(self, expr: RegularExpressionAST) -> Self:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


# Quantifier types
# Power zero {0} {0,0} {,0}
class QuantifierPowerZero(Quantifier):
    @override
    def apply_on_NFA(self, nfa: NFA) -> NFA:
        # R{0} = {ε}
        return NFA.empty_string_language_NFA()

    @override
    def apply_on_expression(self, expr: RegularExpressionAST) -> Self:
        # R{0} = R{0,0} = R{,0} = {ε}
        return EmptyStringAST()

    @override
    def __str__(self) -> str:
        return '{0}'


# Power 1 {1} or {1,1}
class QuantifierPowerOne(Quantifier):
    @override
    def apply_on_NFA(self, nfa: NFA) -> NFA:
        # Do nothing
        return nfa

    @override
    def apply_on_expression(self, expr: RegularExpressionAST) -> Self:
        # R{1} = R{1,1} = R
        return expr

    @override
    def __str__(self) -> str:
        return '{1}'


# Optional ? {0,1} {,1}
class QuantifierOptional(Quantifier):
    @override
    def apply_on_NFA(self, nfa: NFA) -> NFA:
        return NFA.union_empty_string(nfa)

    @override
    def apply_on_expression(self, expr: RegularExpressionAST) -> Self:
        match expr:
            case EmptyLanguageAST() | EmptyStringAST():
                # ε{0,1} = Φ{0,1} = ε
                return EmptyStringAST()
            case QuantifiedAST(inner_expr=inner, quantifier=op):
                match op:
                    case (
                        QuantifierOptional() |  # (R?)? = R?
                        QuantifierKleeneStar() |  # (R*)? = R*
                        QuantifierAtMost()  # (R{,m})? = R{,m}
                    ):
                        return expr
                    case QuantifierKleenePlus():
                        # (R+){0,1} = R*
                        return QuantifiedAST(
                            inner_expr=inner,
                            quantifier=QuantifierKleeneStar()
                        )
                    case _:
                        return QuantifiedAST(
                            inner_expr=inner,
                            quantifier=QuantifierOptional()
                        )
            case _:
                return QuantifiedAST(
                    inner_expr=inner,
                    quantifier=QuantifierOptional()
                )

    @override
    def __str__(self) -> str:
        return '?'


# Ordindary Kleene star * {0,}
class QuantifierKleeneStar(Quantifier):
    @override
    def apply_on_NFA(self, nfa: NFA) -> NFA:
        return NFA.kleene_star(nfa)

    @override
    def apply_on_expression(self, expr: RegularExpressionAST) -> Self:
        match expr:
            case EmptyStringAST() | EmptyLanguageAST():
                # ε* = Φ* = ε
                return EmptyStringAST()
            case QuantifiedAST(inner_expr=inner, quantifier=op):
                match op:
                    case (
                        QuantifierKleeneStar() |
                        QuantifierKleenePlus()
                    ):
                        # (R*)* = R*
                        # (R+)* = R*
                        return expr
                    case _:
                        return QuantifiedAST(
                            inner_expr=inner,
                            quantifier=QuantifierKleeneStar()
                        )
            case _:
                return QuantifiedAST(
                    inner_expr=inner,
                    quantifier=QuantifierKleeneStar()
                )

    @override
    def __str__(self) -> str:
        return '*'


# Kleene plus + {1,}
class QuantifierKleenePlus(Quantifier):
    @override
    def apply_on_NFA(self, nfa: NFA) -> NFA:
        return NFA.kleene_plus(nfa)

    @override
    def __str__(self) -> str:
        return '+'


# Exact {m}
@dataclass(init=True, repr=True, eq=True, frozen=True)
class QuantifierExact(Quantifier):
    exponent: int

    @override
    def apply_on_NFA(self, nfa: NFA) -> NFA:
        return NFA.power(nfa, self.exponent)

    @override
    def __repr__(self) -> str:
        return f'QuantifierExact({self.exponent})'

    @override
    def __str__(self) -> str:
        return f'{{{self.exponent}}}'


# At least {m,}
@dataclass(init=True, repr=True, eq=True, frozen=True)
class QuantifierAtLeast(Quantifier):
    min_count: int

    @override
    def apply_on_NFA(self, nfa: NFA) -> NFA:
        return NFA.at_least_NFA(nfa, self.min_count)

    @override
    def __repr__(self) -> str:
        return f'QuantifierAtLeast({self.min_count})'

    @override
    def __str__(self) -> str:
        return f'{{{self.min_count},}}'


# At most {,m}
@dataclass(init=True, repr=True, eq=True, frozen=True)
class QuantifierAtMost(Quantifier):
    max_count: int

    @override
    def apply_on_NFA(self, nfa: NFA) -> NFA:
        return NFA.at_most_NFA(nfa, self.max_count)

    @override
    def __repr__(self) -> str:
        return f'QuantifierAtMost({self.max_count})'

    @override
    def __str__(self) -> str:
        return f'{{,{self.max_count}}}'


# Bounded {m,n}
@dataclass(init=True, repr=True, eq=True, frozen=True)
class QuantifierBounded(Quantifier):
    min_count: int
    max_count: int

    @override
    def apply_on_NFA(self, nfa: NFA) -> NFA:
        return NFA.bounded_NFA(
            nfa, self.min_count, self.max_count
        )

    @override
    def __repr__(self) -> str:
        return f'QuantifierBounded({self.min_count},{self.max_count})'

    @override
    def __str__(self) -> str:
        return f'{{{self.min_count},{self.max_count}}}'
# Quantifier types end


class QuantifiedAST(RegularExpressionAST):
    def __init__(
        self,
        inner_expr: RegularExpressionAST,
        quantifier: Quantifier
    ):
        self.inner_expr: RegularExpressionAST = inner_expr
        self.quantifier: Quantifier = quantifier

    def __eq__(self, other):
        match other:
            case QuantifiedAST(inner_expr=expr):
                return self.inner_expr == expr
            case _:
                return False

    @override
    def to_NFA(self) -> NFA:
        return self.quantifier.apply_on_NFA(
            self.inner_expr.to_NFA()
        )

    @override
    def __repr__(self) -> str:
        return f'Quantified({repr(self.inner_expr)})'

    @override
    def __str__(self) -> str:
        match self.inner_expr:
            case UnionAST() | ConcatenationAST():
                value = f'({str(self.inner_expr)})*'
            case _:
                value = f'{str(self.inner_expr)}*'
        return f'{value}{self.quantifier}'


class EmptyStringAST(RegularExpressionAST):
    @override
    def to_NFA(self) -> NFA:
        return NFA.empty_string_language_NFA()

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
        return NFA.empty_language_NFA()

    @override
    def __repr__(self) -> str:
        return 'EmptyLanguageAST()'

    @override
    def __str__(self) -> str:
        return 'Φ'


def union(
    a: RegularExpressionAST, b: RegularExpressionAST
) -> RegularExpressionAST:
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


def concatenate(
    a: RegularExpressionAST, b: RegularExpressionAST
) -> RegularExpressionAST:
    match (a, b):
        case (EmptyLanguageAST(), _) | (_, EmptyLanguageAST()):
            # R ∅ = ∅ R = ∅
            return EmptyLanguageAST()
        case (EmptyStringAST(), x) | (x, EmptyStringAST()):
            # R ε = ε R = R
            return x
        case (
            ConcatenationAST(sequence=seq1),
            ConcatenationAST(sequence=seq2)
        ):
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
# Quantified => Primary Quantifier*
# Quantifier => '?' | '*' | '+' | BoundedQuantifier
# BoundedQuantifier => '{' ( NUMBER ',' | ',' NUMER | NUMBER ',' NUMBER ) '}'
# Primary => ε | SYMBOL | Group
# Group => '(' Expression ')'


# Special character to represent empty string
EMPTY_STRING_CHAR = '#'
META_CHARACTERS = {
    EMPTY_STRING_CHAR, '*', '|', '(', ')', '\\',
    '+', '?', '{', '}', ','
}
NUMBERS_PATTERN = compile(r'\d+')
QUANTIFIERS_CHARS = {
    # Characters allowed between { and }
    ',', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
}


class ParseResult:
    def __init__(self, parsed_expression: RegularExpressionAST, error: str):
        self.parsed_expression: RegularExpressionAST = parsed_expression
        self.error: str = error


class RegularExpressionParser:
    def __init__(self, pattern: str):
        self.pattern: str = pattern
        self.pos = 0
        self.current: Token = None
        self.inside_bounded_quantifier = False
        self.generate_next_token()

    def generate_next_token(self):
        begin = self.pos
        current_char = (
            None if self.pos >= len(self.pattern)
            else self.pattern[self.pos]
        )
        match current_char:
            case c if c == EMPTY_STRING_CHAR:
                result = BasicToken(
                    token_type=TokenType.EMPTY_STRING_TOKEN,
                )
            case c if (
                number := NUMBERS_PATTERN.match(self.pattern, self.pos)
                and self.inside_bounded_quantifier
            ):
                result = NumberToken(
                    value=int(number.group()),
                    token_type=TokenType.NUMBER,
                )
            case '*':
                result = BasicToken(
                    token_type=TokenType.KLEENE_STAR,
                )
            case '+':
                result = BasicToken(
                    token_type=TokenType.KLEENE_PLUS,
                )
            case '?':
                result = BasicToken(
                    token_type=TokenType.MARK,
                )
            case '{':
                next_char = (
                    None if self.pos >= len(self.pattern)
                    else self.pattern[self.pos]
                )
                if next_char in QUANTIFIERS_CHARS:
                    result = BasicToken(
                        token_type=TokenType.LEFT_CURLY_BRACE,
                    )
                    self.inside_bounded_quantifier = True
                else:
                    result = SymbolToken(
                        value='{',
                        token_type=TokenType.SYMBOL,
                    )
            case '}' if self.inside_bounded_quantifier:
                result = BasicToken(
                    token_type=TokenType.RIGHT_CURLY_BRACE,
                )
                self.inside_bounded_quantifier = False
            case ',' if self.inside_bounded_quantifier:
                result = BasicToken(
                    token_type=TokenType.COMMA,
                )
            case '|':
                result = BasicToken(
                    token_type=TokenType.UNION_BAR,
                )
            case '(':
                result = BasicToken(
                    token_type=TokenType.LEFT_PARENTHESIS,
                )
            case ')':
                result = BasicToken(
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
                    result = SymbolToken(
                        value=char,
                        token_type=TokenType.SYMBOL,
                    )
                else:
                    raise ValueError('Trailing slash at pattern end')
            case c:
                if self.inside_bounded_quantifier:
                    self.inside_bounded_quantifier = False
                result = SymbolToken(
                    value=c,
                    token_type=TokenType.SYMBOL,
                )
            case None:
                result = None
        self.current = result
        if self.current:
            self.current.pos = begin
            self.pos += 1

    def parse(self) -> RegularExpressionAST:
        if self.current:
            result = self.parse_expression()
            parsed_expression = result.parsed_expression
            error = result.error
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

    def check(self, expected_type: TokenType) -> bool:
        return self.current and self.current.token_type == expected_type

    def consume(self, expected_type: TokenType) -> Token:
        if self.check(expected_type):
            current_copy = deepcopy(self.current)
            self.generate_next_token()
            return current_copy
        return None

    # Expression => Concatenation ( '|' Concatenation )*
    def parse_expression(self) -> ParseResult:
        initial = self.parse_concatenation()
        parsed_expression = initial.parsed_expression
        error = initial.error
        if parsed_expression:
            alternatives: list[RegularExpressionAST] = [parsed_expression]
            while not error and self.consume(TokenType.UNION_BAR):
                right_term = self.parse_concatenation()
                parsed_expression = right_term.parsed_expression
                error = right_term.error
                if error:
                    parsed_expression = None
                elif parsed_expression:
                    match parsed_expression:
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
        initial = self.parse_quantified()
        parsed_expression = initial.parsed_expression
        error = initial.error
        if parsed_expression:
            sequence: list[RegularExpressionAST] = [parsed_expression]
            while not error:
                right_term = self.parse_quantified()
                parsed_expression = right_term.parsed_expression
                error = right_term.error
                if error:
                    parsed_expression = None
                elif parsed_expression:
                    match parsed_expression:
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

    # Quantified => Primary Quantifier*
    def parse_quantified(self) -> ParseResult:
        primary = self.parse_primary()
        parsed_expression = primary.parsed_expression
        error = primary.error
        if parsed_expression:
            while quantifier := self.consume_quantifier():
                parsed_expression = (
                    quantifier.apply_on_expression(parsed_expression)
                )
        else:
            parsed_expression = None
        return ParseResult(parsed_expression, error)

    # Quantifier => '?' | '*' | '+' | BoundedQuantifier
    # BoundedQuantifier =>
    # '{' ( NUMBER ',' | ',' NUMER | NUMBER ',' NUMBER ) '}'
    def consume_quantifier(self) -> Quantifier:
        result = None  # Assuming no quantifier at current position
        if self.consume(TokenType.MARK):
            # ? Optional (zero or one)
            result = QuantifierOptional()
        elif self.consume(TokenType.KLEENE_STAR):
            # * any number (possibly zero)
            result = QuantifierKleeneStar()
        elif self.consume(TokenType.KLEENE_PLUS):
            # + at least one
            result = QuantifierKleenePlus()
        elif self.consume(TokenType.LEFT_CURLY_BRACE):
            # {
            if low := self.consume(TokenType.NUMBER):
                # {m
                if self.check(TokenType.RIGHT_CURLY_BRACE):
                    # {m} exactly m
                    match low.value:
                        case 0:
                            result = QuantifierPowerZero()
                        case 1:
                            result = QuantifierPowerOne()
                        case m:
                            result = QuantifierExact(exponent=m)
                elif self.consume(TokenType.COMMA):
                    # {m,
                    if self.check(TokenType.RIGHT_CURLY_BRACE):
                        # {m,} at least m
                        match low.value:
                            case 0:
                                result = QuantifierKleeneStar()
                            case 1:
                                result = QuantifierKleenePlus()
                            case m:
                                result = QuantifierAtLeast(min_count=m)
                    elif high := self.consume(TokenType.NUMBER):
                        # {m,n} at least m and at most n
                        match (low.value, high.value):
                            case (0, 0):
                                result = QuantifierPowerOne()
                            case (0, 1):
                                result = QuantifierOptional()
                            case (1, 1):
                                result = QuantifierPowerOne()
                            case _:
                                result = QuantifierBounded(
                                    min_count=low.value,
                                    max_count=high.value
                                )
            elif self.consume(TokenType.COMMA):
                # {,
                if self.check(TokenType.RIGHT_CURLY_BRACE):
                    # {,} any number (possibly zero)
                    result = QuantifierKleeneStar()
                elif high := self.consume(TokenType.NUMBER):
                    # {,n} at most n
                    match high.value:
                        case 0:
                            result = QuantifierPowerZero()
                        case 1:
                            result = QuantifierOptional()
                        case _:
                            result = QuantifierAtMost(high.value)
            else:
                # Unexpected token after {
                error = f'\nError in position {self.current.pos}\n'
                error += 'Expected number or , after {\n'
                error += f'{self.pattern}\n'
                error += ' ' * self.current.pos + '^'
                raise ValueError(error)

            # Attempt to consume closing }
            if not self.consume(TokenType.RIGHT_CURLY_BRACE):
                error = f'\nError in position {self.current.pos}\n'
                error += 'Expected }\n'
                error += f'{self.pattern}\n'
                error += ' ' * self.current.pos + '^'
                raise ValueError(error)
        return result

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
        parsed_expression = expr.parsed_expression
        error = expr.error
        if error:
            parsed_expression = None
        elif parsed_expression:
            if self.consume(TokenType.RIGHT_PARENTHESIS):
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
        return str(self.ast)

    @staticmethod
    def from_AST(ast: RegularExpressionAST) -> Self:
        re = RegularExpression(None)
        re.pattern = str(ast)
        re.ast = deepcopy(ast)
        re.nfa = re.ast.to_NFA()
        return re
