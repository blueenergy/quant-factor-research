#!/usr/bin/env python3
"""Alpha158 factors computed on our own OHLCV panel.

Qlib defines Alpha158 as a list of expression strings over ``$close``,
``$open``, ``$high``, ``$low`` and ``$volume``, but evaluating them normally
requires the data to be dumped into qlib's binary format first. That would mean
maintaining a second copy of the price history alongside Mongo, with its own
adjustment and membership semantics to keep in sync. So the expressions are
reproduced here (from ``qlib/contrib/data/loader.py``, MIT licensed) and
evaluated directly against the panel we already have.

Evaluation is vectorised by pivoting the long panel into ``date x symbol``
matrices. Every rolling window then runs down the time axis of a whole matrix
at once, which keeps 158 factors over a million rows tractable, and makes it
structurally impossible for one symbol's history to leak into another's.

One deliberate deviation from qlib: rolling windows use ``min_periods=window``
rather than qlib's ``min_periods=1``. Partial windows would otherwise emit a
60-day factor computed from three days of data, and those values are not
comparable across a cross-section. Callers must supply enough warm-up history
for the longest window instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BASE_FIELDS = ("open", "high", "low", "close", "volume", "vwap")
DEFAULT_WINDOWS = (5, 10, 20, 30, 60)

# Rolling operators in the order qlib emits them, so factor names line up.
ROLLING_OPERATORS = (
    "ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "LOW", "QTLU", "QTLD",
    "RANK", "RSV", "IMAX", "IMIN", "IMXD", "CORR", "CORD", "CNTP", "CNTN",
    "CNTD", "SUMP", "SUMN", "SUMD", "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN",
    "VSUMD",
)


# ===================== Factor definitions =====================
def kbar_features() -> Tuple[List[str], List[str]]:
    """The nine hard-coded candle-shape features."""
    fields = [
        "($close-$open)/$open",
        "($high-$low)/$open",
        "($close-$open)/($high-$low+1e-12)",
        "($high-Greater($open, $close))/$open",
        "($high-Greater($open, $close))/($high-$low+1e-12)",
        "(Less($open, $close)-$low)/$open",
        "(Less($open, $close)-$low)/($high-$low+1e-12)",
        "(2*$close-$high-$low)/$open",
        "(2*$close-$high-$low)/($high-$low+1e-12)",
    ]
    names = ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"]
    return fields, names


def price_features(
    windows: Sequence[int] = (0,),
    features: Sequence[str] = ("OPEN", "HIGH", "LOW", "VWAP"),
) -> Tuple[List[str], List[str]]:
    """Raw price levels normalised by the latest close."""
    fields: List[str] = []
    names: List[str] = []
    for feature in features:
        field = feature.lower()
        for offset in windows:
            fields.append(f"$%s/$close" % field if offset == 0 else f"Ref($%s, %d)/$close" % (field, offset))
            names.append(f"{feature.upper()}{offset}")
    return fields, names


def rolling_features(windows: Sequence[int] = DEFAULT_WINDOWS) -> Tuple[List[str], List[str]]:
    """The 29 rolling operators, each instantiated once per window."""
    close_change = "$close-Ref($close, 1)"
    volume_change = "$volume-Ref($volume, 1)"
    templates: Dict[str, Callable[[int], str]] = {
        "ROC": lambda d: f"Ref($close, {d})/$close",
        "MA": lambda d: f"Mean($close, {d})/$close",
        "STD": lambda d: f"Std($close, {d})/$close",
        "BETA": lambda d: f"Slope($close, {d})/$close",
        "RSQR": lambda d: f"Rsquare($close, {d})",
        "RESI": lambda d: f"Resi($close, {d})/$close",
        "MAX": lambda d: f"Max($high, {d})/$close",
        "LOW": lambda d: f"Min($low, {d})/$close",
        "QTLU": lambda d: f"Quantile($close, {d}, 0.8)/$close",
        "QTLD": lambda d: f"Quantile($close, {d}, 0.2)/$close",
        "RANK": lambda d: f"Rank($close, {d})",
        "RSV": lambda d: f"($close-Min($low, {d}))/(Max($high, {d})-Min($low, {d})+1e-12)",
        "IMAX": lambda d: f"IdxMax($high, {d})/{d}",
        "IMIN": lambda d: f"IdxMin($low, {d})/{d}",
        "IMXD": lambda d: f"(IdxMax($high, {d})-IdxMin($low, {d}))/{d}",
        "CORR": lambda d: f"Corr($close, Log($volume+1), {d})",
        "CORD": lambda d: f"Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), {d})",
        "CNTP": lambda d: f"Mean($close>Ref($close, 1), {d})",
        "CNTN": lambda d: f"Mean($close<Ref($close, 1), {d})",
        "CNTD": lambda d: f"Mean($close>Ref($close, 1), {d})-Mean($close<Ref($close, 1), {d})",
        "SUMP": lambda d: f"Sum(Greater({close_change}, 0), {d})/(Sum(Abs({close_change}), {d})+1e-12)",
        "SUMN": lambda d: f"Sum(Greater(Ref($close, 1)-$close, 0), {d})/(Sum(Abs({close_change}), {d})+1e-12)",
        "SUMD": lambda d: (
            f"(Sum(Greater({close_change}, 0), {d})-Sum(Greater(Ref($close, 1)-$close, 0), {d}))"
            f"/(Sum(Abs({close_change}), {d})+1e-12)"
        ),
        "VMA": lambda d: f"Mean($volume, {d})/($volume+1e-12)",
        "VSTD": lambda d: f"Std($volume, {d})/($volume+1e-12)",
        "WVMA": lambda d: (
            f"Std(Abs($close/Ref($close, 1)-1)*$volume, {d})"
            f"/(Mean(Abs($close/Ref($close, 1)-1)*$volume, {d})+1e-12)"
        ),
        "VSUMP": lambda d: f"Sum(Greater({volume_change}, 0), {d})/(Sum(Abs({volume_change}), {d})+1e-12)",
        "VSUMN": lambda d: f"Sum(Greater(Ref($volume, 1)-$volume, 0), {d})/(Sum(Abs({volume_change}), {d})+1e-12)",
        "VSUMD": lambda d: (
            f"(Sum(Greater({volume_change}, 0), {d})-Sum(Greater(Ref($volume, 1)-$volume, 0), {d}))"
            f"/(Sum(Abs({volume_change}), {d})+1e-12)"
        ),
    }

    fields: List[str] = []
    names: List[str] = []
    for operator in ROLLING_OPERATORS:
        for window in windows:
            fields.append(templates[operator](window))
            names.append(f"{'MIN' if operator == 'LOW' else operator}{window}")
    return fields, names


def alpha158_feature_config(
    windows: Sequence[int] = DEFAULT_WINDOWS,
    price_fields: Sequence[str] = ("OPEN", "HIGH", "LOW", "VWAP"),
) -> Tuple[List[str], List[str]]:
    """Qlib's default Alpha158 set: 9 kbar + 4 price + 29 rolling x 5 windows."""
    fields: List[str] = []
    names: List[str] = []
    for part in (kbar_features(), price_features(features=price_fields), rolling_features(windows)):
        fields += part[0]
        names += part[1]
    return fields, names


# ===================== Factor metadata =====================
FAMILY_KBAR = "K线形态"
FAMILY_PRICE = "价格位置"
FAMILY_TREND = "趋势"
FAMILY_VOLATILITY = "波动"
FAMILY_RANGE = "区间位置"
FAMILY_VOLUME_PRICE = "量价相关"
FAMILY_MOMENTUM = "涨跌结构"
FAMILY_VOLUME = "成交量"

# Keyed by the name prefix that shows up in factor names, so the rolling
# operator ``LOW`` is filed under the ``MIN`` it is renamed to. ``{d}`` is
# substituted with the window.
_FACTOR_META: Dict[str, Tuple[str, str]] = {
    "KMID": (FAMILY_KBAR, "(收盘-开盘)/开盘，实体涨跌幅"),
    "KLEN": (FAMILY_KBAR, "(最高-最低)/开盘，K 线全长"),
    "KMID2": (FAMILY_KBAR, "实体涨跌 / K 线全长"),
    "KUP": (FAMILY_KBAR, "上影线 / 开盘价"),
    "KUP2": (FAMILY_KBAR, "上影线 / K 线全长"),
    "KLOW": (FAMILY_KBAR, "下影线 / 开盘价"),
    "KLOW2": (FAMILY_KBAR, "下影线 / K 线全长"),
    "KSFT": (FAMILY_KBAR, "(2*收盘-最高-最低)/开盘，收盘在当日区间内的偏移"),
    "KSFT2": (FAMILY_KBAR, "收盘偏移 / K 线全长"),
    "OPEN": (FAMILY_PRICE, "开盘价 / 收盘价"),
    "HIGH": (FAMILY_PRICE, "最高价 / 收盘价"),
    "LOW": (FAMILY_PRICE, "最低价 / 收盘价"),
    "VWAP": (FAMILY_PRICE, "成交均价 / 收盘价"),
    "ROC": (FAMILY_TREND, "{d} 日前收盘 / 当前收盘，反向动量"),
    "MA": (FAMILY_TREND, "{d} 日均价 / 当前收盘，均线偏离度"),
    "STD": (FAMILY_VOLATILITY, "{d} 日收盘标准差 / 当前收盘，波动率"),
    "BETA": (FAMILY_TREND, "{d} 日收盘回归斜率 / 当前收盘，趋势强度"),
    "RSQR": (FAMILY_TREND, "{d} 日收盘回归 R²，趋势稳定度"),
    "RESI": (FAMILY_TREND, "{d} 日回归残差 / 当前收盘，偏离趋势线的幅度"),
    "MAX": (FAMILY_RANGE, "{d} 日最高价 / 当前收盘"),
    "MIN": (FAMILY_RANGE, "{d} 日最低价 / 当前收盘"),
    "QTLU": (FAMILY_RANGE, "{d} 日收盘 80% 分位 / 当前收盘"),
    "QTLD": (FAMILY_RANGE, "{d} 日收盘 20% 分位 / 当前收盘"),
    "RANK": (FAMILY_RANGE, "当前收盘在过去 {d} 日中的百分位"),
    "RSV": (FAMILY_RANGE, "{d} 日未成熟随机值，收盘在高低区间中的位置"),
    "IMAX": (FAMILY_RANGE, "{d} 日内最高价距今天数 / {d}"),
    "IMIN": (FAMILY_RANGE, "{d} 日内最低价距今天数 / {d}"),
    "IMXD": (FAMILY_RANGE, "{d} 日内高点与低点出现次序之差 / {d}"),
    "CORR": (FAMILY_VOLUME_PRICE, "{d} 日收盘与 log 成交量的相关系数"),
    "CORD": (FAMILY_VOLUME_PRICE, "{d} 日收盘涨幅与成交量变化率的相关系数"),
    "CNTP": (FAMILY_MOMENTUM, "{d} 日内上涨天数占比"),
    "CNTN": (FAMILY_MOMENTUM, "{d} 日内下跌天数占比"),
    "CNTD": (FAMILY_MOMENTUM, "{d} 日内上涨占比减下跌占比"),
    "SUMP": (FAMILY_MOMENTUM, "{d} 日涨幅之和 / 涨跌幅绝对值之和，类 RSI"),
    "SUMN": (FAMILY_MOMENTUM, "{d} 日跌幅之和 / 涨跌幅绝对值之和"),
    "SUMD": (FAMILY_MOMENTUM, "{d} 日涨幅和与跌幅和之差 / 绝对值之和"),
    "VMA": (FAMILY_VOLUME, "{d} 日均量 / 当日成交量"),
    "VSTD": (FAMILY_VOLUME, "{d} 日成交量标准差 / 当日成交量"),
    "WVMA": (FAMILY_VOLATILITY, "{d} 日涨跌幅加权成交量的标准差 / 均值，量能波动"),
    "VSUMP": (FAMILY_VOLUME, "{d} 日放量之和 / 量变绝对值之和"),
    "VSUMN": (FAMILY_VOLUME, "{d} 日缩量之和 / 量变绝对值之和"),
    "VSUMD": (FAMILY_VOLUME, "{d} 日放量和与缩量和之差 / 绝对值之和"),
}

_WINDOWED_NAME_RE = re.compile(r"^([A-Z]+)(\d+)$")


def _catalog_entry(name: str, expression: str, operator: str, window: Optional[int]) -> Dict[str, Any]:
    family, description = _FACTOR_META[operator]
    return {
        "name": name,
        "expression": expression,
        "family": family,
        "operator": operator,
        "window": window,
        "description": description.format(d=window) if window is not None else description,
    }


def factor_catalog(
    windows: Sequence[int] = DEFAULT_WINDOWS,
    price_fields: Sequence[str] = ("OPEN", "HIGH", "LOW", "VWAP"),
) -> List[Dict[str, Any]]:
    """Describe every Alpha158 factor: expression, family, window, Chinese blurb.

    Built by walking the same three definition builders ``alpha158_feature_config``
    uses, so a factor can never appear in a backtest without a catalog entry.
    The kbar block is handled separately because names like ``KMID2`` end in a
    digit that is not a window.
    """
    entries: List[Dict[str, Any]] = []
    for expression, name in zip(*kbar_features()):
        entries.append(_catalog_entry(name, expression, name, None))
    for expression, name in zip(*price_features(features=price_fields)):
        match = _WINDOWED_NAME_RE.match(name)
        if match is None:  # pragma: no cover - price names always carry an offset
            raise ValueError(f"unexpected price factor name {name!r}")
        entries.append(_catalog_entry(name, expression, match.group(1), int(match.group(2))))
    for expression, name in zip(*rolling_features(windows)):
        match = _WINDOWED_NAME_RE.match(name)
        if match is None:  # pragma: no cover - rolling names always carry a window
            raise ValueError(f"unexpected rolling factor name {name!r}")
        entries.append(_catalog_entry(name, expression, match.group(1), int(match.group(2))))
    return entries


# ===================== Expression parsing =====================
_TOKEN_RE = re.compile(
    r"""\s*(?:
        (?P<number>\d+\.?\d*(?:[eE][-+]?\d+)?)
      | (?P<field>\$[A-Za-z_]\w*)
      | (?P<name>[A-Za-z_]\w*)
      | (?P<op><=|>=|==|!=|[-+*/(),<>?:])
    )""",
    re.VERBOSE,
)


@dataclass(frozen=True)
class Token:
    kind: str
    text: str


def tokenize(expression: str) -> List[Token]:
    tokens: List[Token] = []
    position = 0
    while position < len(expression):
        if expression[position].isspace():
            position += 1
            continue
        match = _TOKEN_RE.match(expression, position)
        if not match:
            raise ValueError(f"cannot tokenize {expression!r} at offset {position}")
        kind = match.lastgroup
        tokens.append(Token(kind, match.group(kind)))
        position = match.end()
    return tokens


class Parser:
    """Recursive-descent parser for qlib's expression grammar.

    Precedence, loosest first: comparison, additive, multiplicative, unary.
    """

    def __init__(self, tokens: Sequence[Token]):
        self.tokens = list(tokens)
        self.position = 0

    def parse(self):
        node = self._comparison()
        if self.position != len(self.tokens):
            raise ValueError(f"unconsumed tokens from {self.tokens[self.position].text!r}")
        return node

    # -- grammar --
    def _comparison(self):
        node = self._additive()
        while self._peek_op() in ("<", ">", "<=", ">=", "==", "!="):
            operator = self._take().text
            node = ("binary", operator, node, self._additive())
        # Alpha101 expressions use the C-style ternary throughout. It binds
        # looser than every comparison and is right-associative, so
        # ``a ? b : c ? d : e`` parses as ``a ? b : (c ? d : e)``.
        if self._peek_op() == "?":
            self._take()
            when_true = self._comparison()
            self._expect(":")
            when_false = self._comparison()
            node = ("ternary", node, when_true, when_false)
        return node

    def _additive(self):
        node = self._multiplicative()
        while self._peek_op() in ("+", "-"):
            operator = self._take().text
            node = ("binary", operator, node, self._multiplicative())
        return node

    def _multiplicative(self):
        node = self._unary()
        while self._peek_op() in ("*", "/"):
            operator = self._take().text
            node = ("binary", operator, node, self._unary())
        return node

    def _unary(self):
        if self._peek_op() == "-":
            self._take()
            return ("negate", self._unary())
        return self._primary()

    def _primary(self):
        token = self._take()
        if token.kind == "number":
            return ("const", float(token.text))
        if token.kind == "field":
            return ("field", token.text[1:].lower())
        if token.kind == "name":
            self._expect("(")
            arguments = []
            if self._peek_op() != ")":
                arguments.append(self._comparison())
                while self._peek_op() == ",":
                    self._take()
                    arguments.append(self._comparison())
            self._expect(")")
            return ("call", token.text, arguments)
        if token.text == "(":
            node = self._comparison()
            self._expect(")")
            return node
        raise ValueError(f"unexpected token {token.text!r}")

    # -- helpers --
    def _peek_op(self):
        if self.position >= len(self.tokens):
            return None
        token = self.tokens[self.position]
        return token.text if token.kind == "op" else None

    def _take(self) -> Token:
        if self.position >= len(self.tokens):
            raise ValueError("unexpected end of expression")
        token = self.tokens[self.position]
        self.position += 1
        return token

    def _expect(self, text: str) -> None:
        token = self._take()
        if token.text != text:
            raise ValueError(f"expected {text!r}, got {token.text!r}")


def parse_expression(expression: str):
    return Parser(tokenize(expression)).parse()


# ===================== Rolling primitives =====================
def _rolling(frame: pd.DataFrame, window: int):
    return frame.rolling(window, min_periods=window)


def _rolling_argfunc(
    frame: pd.DataFrame,
    window: int,
    take_max: bool,
    *,
    prefer_recent: bool = False,
) -> pd.DataFrame:
    """Offset of the window's extreme value, 0 = oldest bar, matching qlib.

    ``prefer_recent`` breaks ties toward the most recent occurrence instead of
    the oldest. Alpha101's ``ts_argmax`` implementations differ here, and the
    difference is visible whenever a stock repeats a high (or low) within the
    window — a flat price on a daily-basic window is enough. The validation
    harness measures which convention Tushare uses.
    """
    values = frame.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan)
    if values.shape[0] >= window:
        from numpy.lib.stride_tricks import sliding_window_view

        strided = sliding_window_view(values, window, axis=0)
        with np.errstate(invalid="ignore"):
            if prefer_recent:
                # argmax over the reversed window finds the newest extreme,
                # then convert back to the oldest-first convention.
                reversed_position = (
                    np.argmax(strided[:, :, ::-1], axis=2)
                    if take_max
                    else np.argmin(strided[:, :, ::-1], axis=2)
                )
                found = ((window - 1) - reversed_position).astype(float)
            else:
                found = (np.argmax(strided, axis=2) if take_max else np.argmin(strided, axis=2)).astype(float)
        # argmax/argmin silently pick position 0 when the window holds a NaN.
        complete = frame.notna().rolling(window, min_periods=window).sum().to_numpy() == window
        found[~complete[window - 1 :]] = np.nan
        out[window - 1 :] = found
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


def _regression_parts(frame: pd.DataFrame, window: int):
    """Rolling OLS of the series on 0..window-1, solved from rolling sums.

    ``sum(i * y_i)`` over a window is rewritten against absolute row positions
    so it becomes two ordinary rolling sums. A ``rolling().apply()`` with a
    Python callback would be correct too, but runs per cell and is unusable on
    a panel of a few million rows.
    """
    x_mean = (window - 1) / 2.0
    sxx = window * (window * window - 1) / 12.0
    positions = np.arange(len(frame), dtype=float)

    sum_y = _rolling(frame, window).sum()
    sum_position_y = _rolling(frame.mul(positions, axis=0), window).sum()
    window_start = positions - window + 1
    sxy = sum_position_y.sub(sum_y.mul(window_start, axis=0)) - window * x_mean * (sum_y / window)

    slope = sxy / sxx
    y_mean = sum_y / window
    intercept = y_mean - slope * x_mean
    return slope, intercept, y_mean


def _slope(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    return _regression_parts(frame, window)[0]


def _rsquare(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    slope, _intercept, _y_mean = _regression_parts(frame, window)
    sxx = window * (window * window - 1) / 12.0
    total_variation = _rolling(frame, window).var(ddof=0) * window
    with np.errstate(invalid="ignore", divide="ignore"):
        return (slope * slope * sxx) / total_variation.replace(0, np.nan)


def _residual(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Residual of the most recent bar against the window's fitted line."""
    slope, intercept, _ = _regression_parts(frame, window)
    return frame - (intercept + slope * (window - 1))


# ===================== Evaluation =====================
def _as_frame(value, template: pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    return pd.DataFrame(float(value), index=template.index, columns=template.columns)


class Evaluator:
    """Evaluate a parsed expression against ``date x symbol`` matrices."""

    def __init__(self, matrices: Dict[str, pd.DataFrame]):
        if not matrices:
            raise ValueError("no base field matrices supplied")
        self.matrices = matrices
        self.template = next(iter(matrices.values()))

    def evaluate(self, node):
        kind = node[0]
        if kind == "const":
            return node[1]
        if kind == "field":
            name = node[1]
            if name not in self.matrices:
                raise KeyError(name)
            return self.matrices[name]
        if kind == "negate":
            return -self._frame(node[1])
        if kind == "binary":
            return self._binary(node[1], node[2], node[3])
        if kind == "ternary":
            return self._ternary(node[1], node[2], node[3])
        if kind == "call":
            return self._call(node[1], node[2])
        raise ValueError(f"unknown node {kind}")

    def _frame(self, node) -> pd.DataFrame:
        return _as_frame(self.evaluate(node), self.template)

    def _binary(self, operator: str, left_node, right_node):
        left, right = self.evaluate(left_node), self.evaluate(right_node)
        if isinstance(left, float) and isinstance(right, float):
            left = _as_frame(left, self.template)
        if operator == "+":
            return left + right
        if operator == "-":
            return left - right
        if operator == "*":
            return left * right
        if operator == "/":
            denominator = _as_frame(right, self.template).replace(0, np.nan)
            return _as_frame(left, self.template) / denominator
        # Comparisons feed straight into Mean(), so emit 1.0/0.0 and keep NaN as NaN.
        left_frame, right_frame = _as_frame(left, self.template), _as_frame(right, self.template)
        comparison = {
            "<": left_frame < right_frame,
            ">": left_frame > right_frame,
            "<=": left_frame <= right_frame,
            ">=": left_frame >= right_frame,
            "==": left_frame == right_frame,
            "!=": left_frame != right_frame,
        }[operator].astype(float)
        return comparison.where(left_frame.notna() & right_frame.notna())

    def _ternary(self, condition_node, true_node, false_node):
        """Elementwise ``condition ? true : false``.

        A NaN condition stays NaN rather than collapsing to the false branch:
        qlib's ``If`` and Alpha101's ``?:`` both treat a missing condition as
        missing, and silently choosing a branch would fabricate observations.
        """
        condition = self._frame(condition_node)
        true_values = _as_frame(self.evaluate(true_node), self.template)
        false_values = _as_frame(self.evaluate(false_node), self.template)
        valid = condition.notna()
        chosen = true_values.where(condition.fillna(0.0).astype(bool), false_values)
        return chosen.where(valid)

    def _call(self, name: str, argument_nodes):
        arguments = [self.evaluate(node) for node in argument_nodes]

        if name == "Abs":
            return self._frame(argument_nodes[0]).abs()
        if name == "Log":
            values = self._frame(argument_nodes[0])
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.log(values.where(values > 0))
        if name in ("Greater", "Less"):
            left = _as_frame(arguments[0], self.template)
            right = _as_frame(arguments[1], self.template)
            combined = np.maximum(left, right) if name == "Greater" else np.minimum(left, right)
            return combined
        if name == "Corr":
            left = _as_frame(arguments[0], self.template)
            right = _as_frame(arguments[1], self.template)
            window = int(arguments[2])
            result = left.rolling(window, min_periods=window).corr(right, pairwise=False)
            return result.replace([np.inf, -np.inf], np.nan)
        if name == "Covariance":
            left = _as_frame(arguments[0], self.template)
            right = _as_frame(arguments[1], self.template)
            window = int(arguments[2])
            result = left.rolling(window, min_periods=window).cov(right, pairwise=False)
            return result.replace([np.inf, -np.inf], np.nan)
        if name == "Sign":
            return np.sign(self._frame(argument_nodes[0]))
        if name == "Delta":
            frame = self._frame(argument_nodes[0])
            return frame - frame.shift(int(arguments[1]))
        if name == "Delay":
            return self._frame(argument_nodes[0]).shift(int(arguments[1]))
        if name in ("TsRank", "Ts_Rank"):
            frame = self._frame(argument_nodes[0])
            window = int(arguments[1])
            return frame.rolling(window, min_periods=window).rank(pct=True)
        if name in ("CSRank", "CrossSectionalRank"):
            return self._frame(argument_nodes[0]).rank(axis=1, pct=True)
        if name in ("Where", "IF"):
            return self._ternary(argument_nodes[0], argument_nodes[1], argument_nodes[2])
        if name == "SignedPower":
            frame = self._frame(argument_nodes[0])
            power = float(arguments[1])
            return np.sign(frame) * np.abs(frame) ** power
        if name in ("TsArgMax", "Ts_ArgMax"):
            frame = self._frame(argument_nodes[0])
            window = int(arguments[1])
            # _rolling_argfunc returns 0 = oldest bar. Alpha101's ts_argmax
            # counts back from today, so 0 = most recent bar. Ties go to the
            # most recent occurrence, matching the Tushare implementation.
            return (window - 1) - _rolling_argfunc(
                frame, window, take_max=True, prefer_recent=True
            )

        frame = _as_frame(arguments[0], self.template)
        if name == "Ref":
            return frame.shift(int(arguments[1]))
        if name == "Quantile":
            return _rolling(frame, int(arguments[1])).quantile(float(arguments[2]))
        if name == "Rank":
            window = int(arguments[1])
            return frame.rolling(window, min_periods=window).rank(pct=True)
        if name == "IdxMax":
            return _rolling_argfunc(frame, int(arguments[1]), take_max=True)
        if name == "IdxMin":
            return _rolling_argfunc(frame, int(arguments[1]), take_max=False)
        if name == "Slope":
            return _slope(frame, int(arguments[1]))
        if name == "Rsquare":
            return _rsquare(frame, int(arguments[1]))
        if name == "Resi":
            return _residual(frame, int(arguments[1]))

        window = int(arguments[1])
        simple = {
            "Mean": "mean",
            "Sum": "sum",
            "Std": "std",
            "Max": "max",
            "Min": "min",
        }
        if name in simple:
            return getattr(_rolling(frame, window), simple[name])()
        raise ValueError(f"unsupported operator {name}")


def build_matrices(
    panel: pd.DataFrame,
    fields: Sequence[str] = BASE_FIELDS,
) -> Dict[str, pd.DataFrame]:
    """Pivot a long panel into ``date x symbol`` matrices per requested field.

    ``fields`` defaults to the OHLCV base set. Callers evaluating Alpha101 or
    fundamental expressions pass their own list (``returns``,
    ``turnover_rate``, ...); a missing field is left out of the result, which
    the evaluator turns into a per-factor skip rather than an approximation.
    """
    requested = tuple(fields)
    available = [field for field in requested if field in panel.columns]
    if not available:
        raise ValueError(f"panel has none of the requested fields {requested}")

    # Reindexing onto the panel's full date and symbol axes keeps every matrix
    # the same shape. Left to itself, pivot_table drops a symbol whose values
    # are all NaN, and the shapes then disagree between fields.
    dates = np.sort(panel["trade_date"].unique())
    symbols = np.sort(panel["symbol"].unique())
    matrices: Dict[str, pd.DataFrame] = {}
    for field in available:
        matrices[field] = (
            panel.pivot_table(index="trade_date", columns="symbol", values=field, aggfunc="last")
            .reindex(index=dates, columns=symbols)
            .astype(float)
        )
    return matrices


def compute_alpha158(
    panel: pd.DataFrame,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    *,
    only: Optional[Sequence[str]] = None,
    on_message: Optional[Callable[[str], None]] = None,
) -> Tuple[pd.DataFrame, List[str], List[Dict[str, str]]]:
    """Attach every computable Alpha158 factor to a long OHLCV panel.

    Returns the panel, the factor names that were produced, and the ones that
    were skipped as ``{"name", "reason"}``. Expressions needing a base field
    the panel lacks (``$vwap`` when no turnover data is stored) are skipped
    rather than silently approximated, and the caller gets to say so: a report
    that only shows 157 of 158 factors is otherwise indistinguishable from a
    bug.

    ``only`` restricts evaluation to the named factors. All 158 on a long
    history costs several gigabytes of float columns, so a follow-up run that
    re-tests a shortlist over more years should name what it needs.
    """
    expressions, names = alpha158_feature_config(windows)
    pairs = list(zip(names, expressions))
    if only is not None:
        wanted = set(only)
        pairs = [(name, expression) for name, expression in pairs if name in wanted]
    return compute_expressions(panel, pairs, fields=BASE_FIELDS, on_message=on_message)


def compute_expressions(
    panel: pd.DataFrame,
    expressions: Sequence[Tuple[str, str]],
    *,
    fields: Optional[Sequence[str]] = None,
    on_message: Optional[Callable[[str], None]] = None,
) -> Tuple[pd.DataFrame, List[str], List[Dict[str, str]]]:
    """Evaluate ``(name, expression)`` pairs against a long panel.

    Same contract as :func:`compute_alpha158`, but the expressions are supplied
    by the caller. This is the entry point for Alpha101 and other factor sets:
    the panel only has to carry the fields the expressions name, and a factor
    whose field is absent is skipped with a reason rather than approximated.

    When ``fields`` is omitted, every column except the two join keys is
    pivoted. Callers on a wide panel should pass an explicit list so the
    matrices stay proportional to what the expressions actually need.
    """
    if fields is None:
        fields = tuple(
            column
            for column in panel.columns
            if column not in ("trade_date", "symbol")
        )
    matrices = build_matrices(panel, fields=fields)
    evaluator = Evaluator(matrices)
    template = evaluator.template

    # Map every panel row to its cell in the date x symbol matrices once, so
    # each factor is scattered back with a single fancy-index instead of a join.
    date_position = {date: i for i, date in enumerate(template.index)}
    symbol_position = {symbol: i for i, symbol in enumerate(template.columns)}
    row_of = panel["trade_date"].map(date_position).to_numpy()
    column_of = panel["symbol"].map(symbol_position).to_numpy()

    computed: Dict[str, np.ndarray] = {}
    skipped: List[Dict[str, str]] = []
    for position, (name, expression) in enumerate(expressions, start=1):
        try:
            values = evaluator.evaluate(parse_expression(expression))
        except KeyError as missing:
            skipped.append({"name": name, "reason": f"needs ${missing.args[0]}"})
            continue
        frame = _as_frame(values, template).reindex(index=template.index, columns=template.columns)
        computed[name] = frame.to_numpy(dtype=float)[row_of, column_of]
        if on_message is not None and position % 25 == 0:
            on_message(f"computed {position}/{len(expressions)} expressions")

    if skipped and on_message is not None:
        detail = ", ".join(f"{item['name']} ({item['reason']})" for item in skipped)
        on_message(f"skipped {len(skipped)} factors: {detail}")

    # Concatenate in one shot; inserting ~157 columns individually fragments the
    # frame badly enough to dominate runtime on a large panel.
    factors = pd.DataFrame(computed, index=panel.index)
    return pd.concat([panel, factors], axis=1), list(computed), skipped
