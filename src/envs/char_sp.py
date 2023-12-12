import io
import itertools
import math
import os
import re
from collections import OrderedDict

import numexpr as ne
import sympy as sp
import torch
from sympy.calculus.util import AccumBounds
from sympy.parsing.sympy_parser import parse_expr
from torch.utils.data import Dataset

SPECIAL_WORDS = ['<s>', '</s>', '<pad>', '(', ')']
SPECIAL_WORDS = SPECIAL_WORDS + \
    [f'<SPECIAL_{i}>' for i in range(len(SPECIAL_WORDS), 10)]


EXP_OPERATORS = {'exp', 'sinh', 'cosh'}
EVAL_SYMBOLS = {'x', 'y', 'z', 'a0', 'a1', 'a2',
                'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9'}
EVAL_VALUES = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 2.1, 3.1]
EVAL_VALUES = EVAL_VALUES + [-x for x in EVAL_VALUES]

TEST_ZERO_VALUES = [0.1, 0.9, 1.1, 1.9]
TEST_ZERO_VALUES = [-x for x in TEST_ZERO_VALUES] + TEST_ZERO_VALUES


def count_nested_exp(s):
    """
    Return the maximum number of nested exponential functions in an infix expression.
    """
    stack = []
    count = 0
    max_count = 0
    for v in re.findall('[+-/*//()]|[a-zA-Z0-9]+', s):
        if v == '(':
            stack.append(v)
        elif v == ')':
            while True:
                x = stack.pop()
                if x in EXP_OPERATORS:
                    count -= 1
                if x == '(':
                    break
        else:
            stack.append(v)
            if v in EXP_OPERATORS:
                count += 1
                max_count = max(max_count, count)
    assert len(stack) == 0
    return max_count


def is_valid_expr(s):
    """
    Check that we are able to evaluate an expression (and that it will not blow in SymPy evaluation).
    """
    s = s.replace('Derivative(f(x),x)', '1')
    s = s.replace('Derivative(1,x)', '1')
    s = s.replace('(E)', '(exp(1))')
    s = s.replace('(I)', '(1)')
    s = s.replace('(pi)', '(1)')
    s = re.sub(
        r'(?<![a-z])(f|g|h|Abs|sign|ln|sin|cos|tan|sec|csc|cot|asin|acos|atan|asec|acsc|acot|tanh|sech|csch|coth|asinh|acosh|atanh|asech|acoth|acsch)\(', '(', s)
    count = count_nested_exp(s)
    if count >= 4:
        return False
    for v in EVAL_VALUES:
        try:
            local_dict = {s: (v + 1e-4 * i)
                          for i, s in enumerate(EVAL_SYMBOLS)}
            value = ne.evaluate(s, local_dict=local_dict).item()
            if not (math.isnan(value) or math.isinf(value)):
                return True
        except (FloatingPointError, ZeroDivisionError, TypeError, MemoryError):
            continue
    return False


def eval_test_zero(eq):
    """
    Evaluate an equation by replacing all its free symbols with random values.
    """
    variables = eq.free_symbols
    assert len(variables) <= 3
    outputs = []
    for values in itertools.product(*[TEST_ZERO_VALUES for _ in range(len(variables))]):
        _eq = eq.subs(zip(variables, values)).doit()
        outputs.append(float(sp.Abs(_eq.evalf())))
    return outputs


class CharSPEnvironment(object):
    SYMPY_OPERATORS = {
        # Elementary functions
        sp.Add: 'add',
        sp.Mul: 'mul',
        sp.Pow: 'pow',
        sp.exp: 'exp',
        sp.log: 'ln',
        sp.Abs: 'abs',
        sp.sign: 'sign',
        # Trigonometric Functions
        sp.sin: 'sin',
        sp.cos: 'cos',
        sp.tan: 'tan',
        sp.cot: 'cot',
        sp.sec: 'sec',
        sp.csc: 'csc',
        # Trigonometric Inverses
        sp.asin: 'asin',
        sp.acos: 'acos',
        sp.atan: 'atan',
        sp.acot: 'acot',
        sp.asec: 'asec',
        sp.acsc: 'acsc',
        # Hyperbolic Functions
        sp.sinh: 'sinh',
        sp.cosh: 'cosh',
        sp.tanh: 'tanh',
        sp.coth: 'coth',
        sp.sech: 'sech',
        sp.csch: 'csch',
        # Hyperbolic Inverses
        sp.asinh: 'asinh',
        sp.acosh: 'acosh',
        sp.atanh: 'atanh',
        sp.acoth: 'acoth',
        sp.asech: 'asech',
        sp.acsch: 'acsch',
        # Derivative
        sp.Derivative: 'derivative',
    }

    # the number of arguments for each operator
    OPERATORS = {
        # Elementary functions
        'add': 2,
        'sub': 2,
        'mul': 2,
        'div': 2,
        'pow': 2,
        'rac': 2,
        'inv': 1,
        'pow2': 1,
        'pow3': 1,
        'pow4': 1,
        'pow5': 1,
        'sqrt': 1,
        'exp': 1,
        'ln': 1,
        'abs': 1,
        'sign': 1,
        # Trigonometric Functions
        'sin': 1,
        'cos': 1,
        'tan': 1,
        'cot': 1,
        'sec': 1,
        'csc': 1,
        # Trigonometric Inverses
        'asin': 1,
        'acos': 1,
        'atan': 1,
        'acot': 1,
        'asec': 1,
        'acsc': 1,
        # Hyperbolic Functions
        'sinh': 1,
        'cosh': 1,
        'tanh': 1,
        'coth': 1,
        'sech': 1,
        'csch': 1,
        # Hyperbolic Inverses
        'asinh': 1,
        'acosh': 1,
        'atanh': 1,
        'acoth': 1,
        'asech': 1,
        'acsch': 1,
        # Derivative
        'derivative': 2,
        # custom functions
        'f': 1,
        'g': 2,
        'h': 3,
    }

    def __init__(self, params):
        self.n_variables = params.n_variables
        self.n_coefficients = params.n_coefficients
        self.int_base = params.int_base

        self.operators = sorted(list(self.OPERATORS.keys()))
        self.elements = [str(i) for i in range(abs(self.int_base))]

        # symbols / elements
        self.constants = ['pi', 'E']
        self.variables = OrderedDict({
            'x': sp.Symbol('x', real=True, nonzero=True),  # , positive=True
            'y': sp.Symbol('y', real=True, nonzero=True),  # , positive=True
            'z': sp.Symbol('z', real=True, nonzero=True),  # , positive=True
            't': sp.Symbol('t', real=True, nonzero=True),  # , positive=True
        })
        self.coefficients = OrderedDict({
            f'a{i}': sp.Symbol(f'a{i}', real=True)
            for i in range(10)
        })

        self.symbols = ['I', 'INT+', 'INT-', 'INT',
                        'FLOAT', '-', '.', '10^', 'Y', "Y'", "Y''"]

        assert 1 <= self.n_variables <= len(self.variables)
        assert 0 <= self.n_coefficients <= len(self.coefficients)
        assert all(v in self.OPERATORS for v in self.SYMPY_OPERATORS.values())

        # vocabulary
        self.words = SPECIAL_WORDS + self.constants + list(self.variables.keys()) + list(
            self.coefficients.keys()) + self.operators + self.symbols + self.elements
        self.index_to_symbol = index_to_symbol = {
            k: v for k, v in enumerate(self.words)}
        self.symbol_to_index = symbol_to_index = {
            v: k for k, v in enumerate(self.words)}
        # where k is the key for each token
        # index_to_symbol[i] returns the i_th token

        assert len(self.words) == len(set(self.words))

        # number of words / indices
        self.n_words = params.n_words = len(self.words)
        self.eos_index = params.eos_index = 0
        self.pad_index = params.pad_index = 1

    def get_symbols(self):
        return self.words

    def idx_to_symbol(self, idx):
        return self.index_to_symbol[idx]

    def symbol_to_idx(self, symbol):
        return self.symbol_to_index[symbol]

    def remove_padding(self, string):
        return string.replace(self.idx_to_symbol(self.pad_index), '')

    def parse_int(self, lst):
        """
        Parse a list that starts with an integer.
        Return the integer value, and the position it ends in the list.
        """
        base = self.int_base
        balanced = self.balanced
        val = 0
        if not (balanced and lst[0] == 'INT' or base >= 2 and lst[0] in ['INT+', 'INT-'] or base <= -2 and lst[0] == 'INT'):
            raise ValueError(
                f"Invalid integer in prefix expression")
        i = 0
        for x in lst[1:]:
            if not (x.isdigit() or x[0] == '-' and x[1:].isdigit()):
                break
            val = val * base + int(x)
            i += 1
        if base > 0 and lst[0] == 'INT-':
            val = -val
        return val, i + 1

    def write_infix(self, token, args):
        """
        Infix representation.
        Convert prefix expressions to a format that SymPy can parse.
        """
        if token == 'add':
            return f'({args[0]})+({args[1]})'
        elif token == 'sub':
            return f'({args[0]})-({args[1]})'
        elif token == 'mul':
            return f'({args[0]})*({args[1]})'
        elif token == 'div':
            return f'({args[0]})/({args[1]})'
        elif token == 'pow':
            return f'({args[0]})**({args[1]})'
        elif token == 'rac':
            return f'({args[0]})**(1/({args[1]}))'
        elif token == 'abs':
            return f'Abs({args[0]})'
        elif token == 'inv':
            return f'1/({args[0]})'
        elif token == 'pow2':
            return f'({args[0]})**2'
        elif token == 'pow3':
            return f'({args[0]})**3'
        elif token == 'pow4':
            return f'({args[0]})**4'
        elif token == 'pow5':
            return f'({args[0]})**5'
        elif token in ['sign', 'sqrt', 'exp', 'ln', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'asin', 'acos', 'atan', 'acot', 'asec', 'acsc', 'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch', 'asinh', 'acosh', 'atanh', 'acoth', 'asech', 'acsch']:
            return f'{token}({args[0]})'
        elif token == 'derivative':
            return f'Derivative({args[0]},{args[1]})'
        elif token == 'f':
            return f'f({args[0]})'
        elif token == 'g':
            return f'g({args[0]},{args[1]})'
        elif token == 'h':
            return f'h({args[0]},{args[1]},{args[2]})'
        elif token.startswith('INT'):
            return f'{token[-1]}{args[0]}'
        else:
            return token

    def _prefix_to_infix(self, expr):
        """
        Parse an expression in prefix mode, and output it in either:
          - infix mode (returns human readable string)
          - develop mode (returns a dictionary with the simplified expression)
        """
        if len(expr) == 0:
            raise ValueError("Empty expression")
        t = expr[0]
        if t in self.operators:
            args = []
            l1 = expr[1:]
            for _ in range(self.OPERATORS[t]):
                i1, l1 = self._prefix_to_infix(l1)
                args.append(i1)
            return self.write_infix(t, args), l1
        elif t in self.variables or t in self.coefficients or t in self.constants or t == 'I':
            return t, expr[1:]
        else:
            val, i = self.parse_int(expr)
            return str(val), expr[i:]

    def prefix_to_infix(self, expr):
        """
        Prefix to infix conversion.
        """
        p, r = self._prefix_to_infix(expr)
        if len(r) > 0:
            raise ValueError(
                f"Incorrect prefix expression \"{expr}\". \"{r}\" was not parsed.")
        return f'({p})'

    def infix_to_sympy(self, infix, no_rewrite=False):
        """
        Convert an infix expression to SymPy.
        """
        if not is_valid_expr(infix):
            raise ValueError("Invalid expression")
        expr = parse_expr(infix, evaluate=True, local_dict=self.local_dict)
        if expr.has(sp.I) or expr.has(AccumBounds):
            raise ValueError("Invalid expression")
        if not no_rewrite:
            expr = self.rewrite_sympy_expr(expr)
        return expr


class CharSPDataset(Dataset):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        return self.src[index], self.tgt[index]


class DataParser():
    def __init__(self, env, params, path):
        # data parameters
        self.path = path
        self.train_num = params.train_num
        self.batch_size = params.batch_size

        # environment parameters
        self.env = env
        self.pad_index = params.pad_index
        self.max_seq_len = params.max_seq_len
        self.int_base = params.int_base

        self.inputs = []
        self.outputs = []

        assert os.path.isfile(path), f'Invalid path: {path}'

    def tokenize(self, string):
        symbol_indices = [
            self.env.symbol_to_idx(s) for s in string.split(' ')
        ]

        for _ in range(len(symbol_indices), self.max_seq_len):
            symbol_indices.append(self.pad_index)
        return torch.tensor(symbol_indices)

    def valid_data(self, string) -> bool:
        if len(string) > self.max_seq_len:
            return False
        # check if int greater than int_base
        for s in string.split(' '):
            if s.isdigit() and int(s) >= self.int_base:
                return False

        return True

    def parse(self) -> CharSPDataset:
        with io.open(self.path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            for num, line in enumerate(lines):
                if num == self.train_num:
                    break

                # ignore characters before the first pipe
                line = line[line.find('|')+1:].strip()

                # split the input and output
                input, output = line.split('\t')

                if not self.valid_data(input) or not self.valid_data(output):
                    continue

                # tokenize the input and output
                input = self.tokenize(input)
                output = self.tokenize(output)

                # save the input and output
                self.inputs.append(input)
                self.outputs.append(output)

        self.dataset = CharSPDataset(self.inputs, self.outputs)

        return self.dataset
