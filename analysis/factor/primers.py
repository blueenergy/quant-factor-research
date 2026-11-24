"""Plain-language primers for the factor registry.

A *primer* explains what a factor is trying to measure, how to read its
value, a worked numeric example, and when it is most likely to lie. It is
documentation, not state: the registry keeps the verdict and the evidence
ledger, this file keeps the explanation.

Why this lives in code and not in Mongo
---------------------------------------
The registry already follows one pattern for definitions: the source of truth
is a module here (``quant-scorer/factor_panel/definitions.py`` ->
``finance.factor_definitions``; ``analysis/factor/alpha158_features.py`` ->
``quant_trading.factor_catalog``) and Mongo holds a materialised copy that
the API reads. Primers follow the same pattern: ``tools/sync_factor_primers.py``
expands this file into ``quant_trading.factor_registry.primer`` and the
dashboard reads that copy. Prose edited by hand in Mongo has no diff, no
review and no rollback, so it does not belong there.

Primer keys
-----------
``OPERATOR_PRIMERS``  an alpha158 operator template, keyed by the operator
                      name (``STD``, ``KMID``, ...), shared by every window of
                      that operator. ``{d}`` is replaced by the factor's
                      window.
``FACTOR_PRIMERS``    anything else (alpha8, fm1, vault, group, alpha101),
                      keyed by the registry document's ``_id``.
``FAMILY_PRIMERS``    fallback keyed by the registry document's ``family``,
                      used when a factor has no primer of its own.

Every primer carries the same four fields plus optional references:

    intuition   what the factor is trying to capture, in one sentence
    reading     how to read the number: sign, typical magnitude, what is good
    example     a worked numeric example
    caveats     when it is most likely to mislead
    refs        authoritative links (papers, official docs)

A small number of factors also carry a ``detail`` list of
``{"heading", "body"}`` sections: the long-form write-up that explains the
caliber choice, the role the factor plays in this system, what the measured
evidence actually said, and the misuse patterns. It is written for the
factors that are actually in use, not for all 200.
"""

from __future__ import annotations

from typing import Any, Dict, List

# --------------------------------------------------------------------------
# References. Every URL here was resolved and checked on 2026-11-23.
# --------------------------------------------------------------------------
REF_QLIB_DOC = {
    "title": "Qlib 官方文档：数据层与 Alpha158 数据集",
    "url": "https://qlib.readthedocs.io/en/latest/component/data.html",
}
REF_QLIB_SRC = {
    "title": "Qlib 源码：ALPHA101 / ALPHA158 表达式原文",
    "url": "https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py",
}
REF_ALPHA101_PAPER = {
    "title": "Kakushadze, 101 Formulaic Alphas（arXiv:1601.00991）",
    "url": "https://arxiv.org/abs/1601.00991",
}
REF_MOMENTUM = {
    "title": "Jegadeesh & Titman (1993), Returns to Buying Winners and Selling Losers",
    "url": "https://doi.org/10.1111/j.1540-6261.1993.tb04702.x",
}
REF_CAPM = {
    "title": "Sharpe (1964), Capital Asset Prices: A Theory of Market Equilibrium",
    "url": "https://doi.org/10.1111/j.1540-6261.1964.tb02865.x",
}
REF_ARCH = {
    "title": "Engle (1982), Autoregressive Conditional Heteroscedasticity",
    "url": "https://doi.org/10.2307/1912773",
}
REF_ILLIQ = {
    "title": "Amihud (2002), Illiquidity and stock returns: cross-section and time-series",
    "url": "https://doi.org/10.1016/S1386-4181(01)00024-6",
}
REF_FF93 = {
    "title": "Fama & French (1993), Common risk factors in the returns on stocks and bonds",
    "url": "https://doi.org/10.1016/0304-405X(93)90023-5",
}
REF_FF5 = {
    "title": "Fama & French (2015), A Five-Factor Asset Pricing Model（NBER w21577）",
    "url": "https://www.nber.org/papers/w21577",
}
REF_QFACTOR = {
    "title": "Hou, Xue & Zhang (2015), Digesting Anomalies: An Investment Approach（NBER w20982）",
    "url": "https://www.nber.org/papers/w20982",
}
REF_GROSSPROFIT = {
    "title": "Novy-Marx (2013), The other side of value: The gross profitability premium",
    "url": "https://doi.org/10.1016/j.jfineco.2013.01.003",
}
REF_NEWEY_WEST = {
    "title": "Newey & West (1987), A Simple, Positive Semi-Definite HAC Covariance Matrix",
    "url": "https://doi.org/10.2307/1913610",
}
REF_MULTIPLE_TESTING = {
    "title": "Harvey, Liu & Zhu, ... and the Cross-Section of Expected Returns（NBER w20592）",
    "url": "https://www.nber.org/papers/w20592",
}
REF_TUSHARE = {
    "title": "Tushare Pro 接口文档（财务与行情字段口径）",
    "url": "https://tushare.pro/document/2",
}
REF_SSE = {
    "title": "上海证券交易所（交易规则、披露规则）",
    "url": "https://www.sse.com.cn/",
}
REF_SZSE = {
    "title": "深圳证券交易所（交易规则、披露规则）",
    "url": "https://www.szse.cn/",
}
REF_CSINDEX = {
    "title": "中证指数有限公司（指数编制与成分）",
    "url": "https://www.csindex.com.cn/",
}

# --------------------------------------------------------------------------
# alpha158: one primer per operator. ``{d}`` is the window.
# --------------------------------------------------------------------------
OPERATOR_PRIMERS: Dict[str, Dict[str, Any]] = {
    # ---- K 线形态 ----
    "KMID": {
        "intuition": "当天收盘比开盘涨了多少，量的是这根 K 线里多方赢了多少。",
        "reading": "A 股个股典型值 ±0.5%~3%；正数代表收阳。绝对值越大说明当天多空分歧越大。",
        "example": "开盘 10.00、收盘 10.20，KMID = (10.20-10.00)/10.00 = +2.0%。",
        "caveats": "单日信息噪声极大，单独几乎没有预测力；A 股 T+1，当日买入不可卖，"
                   "「今天的 K 线」对次日开盘没有直接可交易含义。",
        "refs": [REF_QLIB_DOC],
    },
    "KLEN": {
        "intuition": "当天波动有多大（振幅相对开盘价），量的是不安分的程度。",
        "reading": "A 股个股典型 2%~6%；超过 7% 属于异动，需要单独看事件。",
        "example": "开盘 10.00、最高 10.60、最低 9.80，KLEN = (10.60-9.80)/10.00 = 8.0%。",
        "caveats": "振幅大是风险特征而不是方向特征，常与波动率类因子高度相关，"
                   "同时放进模型容易重复计权。",
        "refs": [REF_QLIB_DOC],
    },
    "KMID2": {
        "intuition": "把方向放进振幅里衡量：实体涨跌占整根 K 线全长的比例。",
        "reading": "取值在 -1 到 +1 之间；接近 +1 表示几乎单边收在最高，接近 0 是十字星。",
        "example": "开盘 10.00、最高 10.60、最低 9.80、收盘 10.50，"
                   "KMID2 = 0.50 / 0.80 = +0.625。",
        "caveats": "与 KSFT2（收盘在区间中的位置）几乎同源，两者的横截面相关性极高。",
        "refs": [REF_QLIB_DOC],
    },
    "KUP": {
        "intuition": "上影线占开盘价的比例，量的是冲高之后被打回来的上方卖压。",
        "reading": "常见 0~5%；长上影（>5%）意味着盘中冲高回落，浮筹被打掉。",
        "example": "开盘 10.00、最高 10.50、收盘 10.10，KUP = (10.50-10.10)/10.00 = 4.0%。",
        "caveats": "单根 K 线的影线对次日方向的预测力很弱，"
                   "它更多是「日内博弈结构」的描述而不是信号。",
        "refs": [REF_QLIB_DOC],
    },
    "KUP2": {
        "intuition": "上影线占整根 K 线全长的比例，比 KUP 更不受开盘价影响。",
        "reading": "0~1；接近 1 表示这是一根几乎全是上影的 K 线。",
        "example": "开盘 10.00、最高 10.60、最低 9.80、收盘 10.50，KUP2 = 0.10 / 0.80 = 0.125。",
        "caveats": "与 KUP 高度相关，两个同时用几乎等于把同一个信息说两遍。",
        "refs": [REF_QLIB_DOC],
    },
    "KLOW": {
        "intuition": "下影线占开盘价的比例，量的是杀跌之后被买回来的下方承接力。",
        "reading": "常见 0~5%；长下影常出现在急跌企稳的日子。",
        "example": "开盘 10.00、最低 9.60、收盘 9.90，KLOW = (9.90-9.60)/10.00 = 3.0%。",
        "caveats": "「长下影 = 见底」是事后叙述，次日方向并不确定；"
                   "在单边下跌里长下影往往只是下跌中继。",
        "refs": [REF_QLIB_DOC],
    },
    "KLOW2": {
        "intuition": "下影线占整根 K 线全长的比例。",
        "reading": "0~1；和 KLOW 一样看「这根 K 线有多少是下影」。",
        "example": "开盘 10.00、最高 10.10、最低 9.60、收盘 9.90，KLOW2 = 0.30 / 0.50 = 0.60。",
        "caveats": "与 KLOW 高度相关；alpha158 内部冗余度较高，不要因为「多」就都留下。",
        "refs": [REF_QLIB_DOC],
    },
    "KSFT": {
        "intuition": "收盘价站在当日最高最低区间中点的哪一侧（经典收盘位置指标）。",
        "reading": "接近 +1 表示收在当日最高，接近 -1 表示收在当日最低，0 表示收在区间正中间。",
        "example": "最高 10.60、最低 9.80、收盘 10.50，KSFT = (2*10.50-10.60-9.80)/10.00 = +0.60%。",
        "caveats": "和 KMID2 信息重叠；单独看「今天收得好不好」无法区分是趋势还是噪声。",
        "refs": [REF_QLIB_DOC],
    },
    "KSFT2": {
        "intuition": "收盘位置占整根 K 线全长的比例（KSFT 的归一化版本）。",
        "reading": "-1 到 +1，含义同 KSFT，但不受开盘价高低的影响。",
        "example": "最高 10.60、最低 9.80、收盘 10.50，KSFT2 = 0.60 / 0.80 = 0.75。",
        "caveats": "与 KMID2 近乎同源，二选一即可。",
        "refs": [REF_QLIB_DOC],
    },
    # ---- 价格位置 ----
    "OPEN": {
        "intuition": "开盘价相对收盘价的位置，反过来读就是「今天从开盘到收盘涨了还是跌了」。",
        "reading": "典型 0.98~1.02；大于 1 表示收盘低于开盘（当天跌）。",
        "example": "开盘 10.00、收盘 9.80，OPEN = 10.00/9.80 = 1.020。",
        "caveats": "与 KMID 是一一对应的单调变换，同时放两个等于重复计权。",
        "refs": [REF_QLIB_DOC],
    },
    "HIGH": {
        "intuition": "当日最高价相对收盘价的距离，量的是「今天冲上去多少」。",
        "reading": "典型 1.00~1.05；值大说明当天曾有明显的冲高。",
        "example": "最高 10.30、收盘 10.00，HIGH = 1.030。",
        "caveats": "只看「冲高幅度」会误判方向：冲高之后是回落还是封住，差别很大。",
        "refs": [REF_QLIB_DOC],
    },
    "LOW": {
        "intuition": "当日最低价相对收盘价的距离，量的是「今天杀下去多少」。",
        "reading": "典型 0.95~1.00；值小（离 1 远）说明当天跌幅深。",
        "example": "最低 9.50、收盘 10.00，LOW = 0.950。",
        "caveats": "与 HIGH 一样是「幅度」不是「方向」，两者一起看才是当天的振幅结构。",
        "refs": [REF_QLIB_DOC],
    },
    "VWAP": {
        "intuition": "当日成交均价相对收盘价的位置，量的是「收盘时持仓者的平均成本比现价高还是低」。",
        "reading": "大于 1 表示收盘价低于当日平均成本，当天多数买入者是亏的；小于 1 反之。",
        "example": "当日成交均价 10.10、收盘 10.00，VWAP = 1.010。",
        "caveats": "均价里混着当日的全部成交，无法区分主动买入还是被动卖出；"
                   "只能当作「当天筹码成本」的一个粗略刻度。",
        "refs": [REF_QLIB_DOC],
    },
    # ---- 趋势 ----
    "ROC": {
        "intuition": "当前价与 {d} 日前价格之比，写成倒数形式，所以它是反向动量：值大 = 过去跌得多。",
        "reading": "典型 0.8~1.3；ROC 大表示过去 {d} 日跌幅大。",
        "example": "当前 10.00、{d} 日前 12.00，ROC = 12.00/10.00 = 1.20（即过去跌了 16.7%）。",
        "caveats": "方向容易搞反：这个因子名字里的「ROC」是涨跌幅，但这里被取过倒数。"
                   "和 momentum_{d} 是严格单调反向关系，不要两个一起用。",
        "refs": [REF_QLIB_DOC, REF_MOMENTUM],
    },
    "MA": {
        "intuition": "当前价相对 {d} 日均线的位置，量的是「现价站在均线上方还是下方」。",
        "reading": "典型 0.95~1.05；大于 1 表示现价在均线上方。",
        "example": "当前 10.00、过去 {d} 日均价 9.70，MA = 9.70/10.00 = 0.970。",
        "caveats": "与动量类因子高度相关；在单边上涨行情里几乎所有股票都大于 1，横截面区分度会塌掉。",
        "refs": [REF_QLIB_DOC],
    },
    "STD": {
        "intuition": "过去 {d} 日收盘价的标准差（再除以收盘价做归一），量的是这段时间价格抖得有多厉害。",
        "reading": "A 股个股典型 2%~6%（日频）；值大 = 高波动股，小盘和题材股普遍更高。",
        "example": "过去 {d} 日收盘价在 10 元附近波动 ±0.3（约收盘价的 3%）⇒ STD{d} ≈ 3%。",
        "caveats": "与 fm1 的 volatility_20d 同源（只是没有年化）；"
                   "低波动组合在 A 股长期有效，但成长行情里会被明显压制。",
        "refs": [REF_QLIB_DOC, REF_ARCH],
    },
    "BETA": {
        "intuition": "对最近 {d} 日价格做时间回归得到的斜率（再除以收盘价），量的是趋势有多陡。",
        "reading": "典型 -0.05~+0.05；正值表示这段时间价格整体在上行，且比动量更强调「加速」。",
        "example": "回归斜率 +0.15 元/日、当前价 10.00 ⇒ BETA{d} = 0.15/10.00 = +0.015。",
        "caveats": "回归斜率对跳空和涨跌停极敏感；A 股有涨跌停板，斜率分布厚尾，小样本上不稳定。",
        "refs": [REF_QLIB_DOC],
    },
    "RSQR": {
        "intuition": "同一个回归的 R²，量的是「这条趋势干不干净」——同样的斜率，稳步上行和来回震荡的 R² 差很多。",
        "reading": "0~1；一般大于 0.5 才算有明确趋势，小于 0.2 基本是噪声。",
        "example": "价格几乎沿直线上涨时 RSQR{d} 接近 0.95；锯齿走势则接近 0.1。",
        "caveats": "R² 高不代表斜率大——横盘中的微小波动也能有很高的 R²，必须和 BETA 一起读。",
        "refs": [REF_QLIB_DOC],
    },
    "RESI": {
        "intuition": "当前价偏离自己趋势线的幅度（回归残差），是「短期超买超卖 / 均值回复」的度量。",
        "reading": "典型 ±0.1；正值表示现价在趋势线上方（短期偏贵）。",
        "example": "趋势线预测 9.50、实际收 10.00 ⇒ RESI{d} = 0.50/10.00 = +5%。",
        "caveats": "本质是反转型因子：在强趋势行情里会持续亏损（趋势市 vs 均值回复 regime 切换）。",
        "refs": [REF_QLIB_DOC],
    },
    # ---- 波动 ----
    "WVMA": {
        "intuition": "用成交量加权的价格波动的标准差（再除以均值），量的是「有量参与时价格抖得有多厉害」。",
        "reading": "典型 0.01~0.05；与 STD 同为波动度量，但放大了放量时的价格冲击。",
        "example": "放量日振幅 4%、缩量日振幅 1%，加权后标准差会明显高于普通波动率。",
        "caveats": "它同时含有「波动」和「量」两种信息，容易和量价相关类因子在组合里互相抵消。",
        "refs": [REF_QLIB_DOC],
    },
    # ---- 区间位置 ----
    "MAX": {
        "intuition": "过去 {d} 日最高价相对当前收盘价的位置，量的是「离前高还有多远」。",
        "reading": "典型 1.00~1.30；接近 1 表示现在就在阶段高位。",
        "example": "过去 {d} 日最高价 12.00、当前 10.00 ⇒ MAX{d} = 1.20。",
        "caveats": "前高往往是套牢盘密集区，价格靠近时容易受阻；这是叙事而不是稳定的统计规律。",
        "refs": [REF_QLIB_DOC],
    },
    "MIN": {
        "intuition": "过去 {d} 日最低价相对当前收盘价的位置，量的是「离前低还有多远」。",
        "reading": "典型 0.70~1.00；接近 1 表示现在就在阶段低位。",
        "example": "过去 {d} 日最低价 8.00、当前 10.00 ⇒ MIN{d} = 0.80。",
        "caveats": "与 MAX 一起构成「价格在区间里的位置」，单用一个信息不完整。",
        "refs": [REF_QLIB_DOC],
    },
    "QTLU": {
        "intuition": "过去 {d} 日收盘价的 80% 分位相对当前价的位置，量的是「现价离区间上沿多近」。",
        "reading": "典型 1.00~1.20。",
        "example": "过去 {d} 日收盘价 80% 分位为 11.00、当前 10.00 ⇒ QTLU{d} = 1.10。",
        "caveats": "分位数对窗口和样本量敏感；不同窗口的 QTLU 不可直接比较。",
        "refs": [REF_QLIB_DOC],
    },
    "QTLD": {
        "intuition": "过去 {d} 日收盘价的 20% 分位相对当前价的位置，量的是「离区间下沿多近」。",
        "reading": "典型 0.80~1.00。",
        "example": "过去 {d} 日收盘价 20% 分位为 9.00、当前 10.00 ⇒ QTLD{d} = 0.90。",
        "caveats": "与 QTLU 互补，合起来才是完整的区间位置。",
        "refs": [REF_QLIB_DOC],
    },
    "RANK": {
        "intuition": "当前收盘价在过去 {d} 日中的百分位，量的是「现在处在自己的区间什么位置」。",
        "reading": "0~1，0.95 表示接近 {d} 日最高；这是无量纲的，比 MAX/MIN 更好跨股票比较。",
        "example": "当前价在窗口内由低到高排到 90% 分位 ⇒ RANK{d} ≈ 0.90。",
        "caveats": "和 K 线位置类因子（KSFT/RANK）分属不同时间尺度：RANK 看 {d} 日，KMID/KSFT 看今天。",
        "refs": [REF_QLIB_DOC],
    },
    "RSV": {
        "intuition": "（收盘 - {d} 日最低）/（{d} 日最高 - {d} 日最低），最早出现在威科夫 RSV 里的随机指标。",
        "reading": "0~1；接近 1 表示收在区间顶部，接近 0 表示收在区间底部。",
        "example": "收盘 10.50、窗口内最低 9.80、最高 10.60 ⇒ RSV{d} = 0.70/0.80 = 0.875。",
        "caveats": "「超买就该跌」在强趋势里会连续失效；它只是价格位置，不是反转信号本身。",
        "refs": [REF_QLIB_DOC],
    },
    "IMAX": {
        "intuition": "过去 {d} 日里最高价出现得有多早（距离今天多少天，除以 {d}），量的是「高点是新的还是老的」。",
        "reading": "0~1；接近 0 表示最高点就在最近一两天（趋势新鲜）。",
        "example": "最高价出现在窗口内第 2 天 ⇒ IMAX{d} = 2/{d}（{d}=20 时约 0.10）。",
        "caveats": "它是「事件发生的时间」而不是幅度，单独用几乎没有经济含义，要和幅度类因子配合。",
        "refs": [REF_QLIB_DOC],
    },
    "IMIN": {
        "intuition": "过去 {d} 日里最低价出现得有多早。",
        "reading": "0~1；接近 0 表示低点就在最近（近期探底）。",
        "example": "最低价出现在窗口内第 1 天 ⇒ IMIN{d} = 1/{d}（{d}=20 时约 0.05）。",
        "caveats": "与 IMAX 一样是时点信息，单独不成信号。",
        "refs": [REF_QLIB_DOC],
    },
    "IMXD": {
        "intuition": "区间内高点与低点出现次序之差（归一化），量的是「这轮是先冲高还是先探底」的顺序结构。",
        "reading": "-1~+1；正数偏表示高点在后（先跌后涨），负数偏表示低点在前。",
        "example": "最低价在第 3 天、最高价在第 17 天 ⇒ IMXD{d} = (17-3)/{d}（{d}=20 时约 +0.70）。",
        "caveats": "次序信息在震荡市里非常不稳定，跨 regime 迁移能力弱。",
        "refs": [REF_QLIB_DOC],
    },
    # ---- 量价相关 ----
    "CORR": {
        "intuition": "过去 {d} 日收盘价与成交量对数的相关系数，量的是「价和量是不是同向走」。",
        "reading": "-1~+1；正 = 放量上涨、缩量下跌的健康结构，负 = 背离。",
        "example": "过去 {d} 日价涨时量也涨、价跌时量也跌 ⇒ CORR{d} 约 +0.5；若放量滞涨则接近 0 或负。",
        "caveats": "相关性对异常日（停牌复牌、公告跳空）非常敏感；A 股散户主导下量价关系经常阶段性反转。",
        "refs": [REF_QLIB_DOC],
    },
    "CORD": {
        "intuition": "过去 {d} 日「价格变动」与「成交量变动率」的相关系数，量的是价的变动是否伴随量的变动。",
        "reading": "-1~+1；正 = 量价同步加速。",
        "example": "连续上涨且量能同步放大时 CORD20 为正；价涨量缩则接近 0 或负。",
        "caveats": "与 CORR 同族但不是同一件事（CORR 用价与量水平，CORD 用价与量变化率），"
                   "两者常常同向，选一个即可。",
        "refs": [REF_QLIB_DOC],
    },
    # ---- 涨跌结构 ----
    "CNTP": {
        "intuition": "过去 {d} 日中上涨天数的占比，量的是「上涨的连贯性」。",
        "reading": "0~1；持续上涨阶段接近 0.8~1.0。",
        "example": "过去 {d} 日里上涨天数占比 0.70 ⇒ CNTP{d} = 0.70。",
        "caveats": "它丢掉了幅度信息：天天小涨和偶尔大涨在这里是一样的，"
                   "要配合 SUMD 一起看。",
        "refs": [REF_QLIB_DOC],
    },
    "CNTN": {
        "intuition": "过去 {d} 日中下跌天数的占比。",
        "reading": "0~1；和 CNTP 互补（两者相加为 1，涨跌停不算平盘）。",
        "example": "过去 {d} 日里下跌天数占比 0.60 ⇒ CNTN{d} = 0.60。",
        "caveats": "与 CNTP 严格互补，同时用等于只用一个加一个冗余。",
        "refs": [REF_QLIB_DOC],
    },
    "CNTD": {
        "intuition": "上涨天数占比减去下跌天数占比，把「涨的天数比跌的多多少」压成一个数。",
        "reading": "-1~+1；正 = 涨多跌少，0 = 涨跌各半。",
        "example": "上涨天数占比 0.70、下跌 0.30 ⇒ CNTD{d} = +0.40。",
        "caveats": "同样是频率而非幅度指标；单边行情里它会长期贴顶，横截面区分度下降。",
        "refs": [REF_QLIB_DOC],
    },
    "SUMP": {
        "intuition": "过去 {d} 日涨幅之和除以涨跌幅绝对值之和，量的是「上涨在波动中占的比重」，类似 RSI。",
        "reading": "0~1；接近 1 = 几乎只有涨没有跌，接近 0.5 = 涨跌平衡。",
        "example": "涨跌幅绝对值合计中上涨贡献 70% ⇒ SUMP{d} = 0.70。",
        "caveats": "分母很小时该比率会剧烈跳动（新股、长期停牌复牌后尤其明显），需要设最小样本量。",
        "refs": [REF_QLIB_DOC],
    },
    "SUMN": {
        "intuition": "过去 {d} 日跌幅之和占涨跌幅绝对值之和的比重。",
        "reading": "0~1；与 SUMP 互补（SUMP + SUMN = 1）。",
        "example": "涨跌幅绝对值合计中下跌贡献 30% ⇒ SUMN{d} = 0.30。",
        "caveats": "与 SUMP 严格互补，同时用只多一份冗余。",
        "refs": [REF_QLIB_DOC],
    },
    "SUMD": {
        "intuition": "涨幅和与跌幅和之差，再除以绝对值之和，把「涨的劲头减跌的劲头」压成一个数。",
        "reading": "-1~+1；正 = 涨得更猛。",
        "example": "上涨占总波动的 70%、下跌占 30% ⇒ SUMD{d} = +0.40。",
        "caveats": "相对 CNTD，这个版本带上了幅度信息，两者作用相近，取其一更干净。",
        "refs": [REF_QLIB_DOC],
    },
    # ---- 成交量 ----
    "VMA": {
        "intuition": "过去 {d} 日平均成交量除以当日成交量，量的是「今天相对近期是放量还是缩量」。",
        "reading": "大于 1 表示今天比近期平均量小（缩量），小于 1 表示放量。",
        "example": "过去 {d} 日日均量 1000 万股、今天 1500 万股 ⇒ VMA{d} = 0.67。",
        "caveats": "绝对量本身没有意义，跨股票不可比；它只反映「相对自己的近期」。",
        "refs": [REF_QLIB_DOC],
    },
    "VSTD": {
        "intuition": "过去 {d} 日成交量的标准差除以当日成交量，量的是「量能本身有多不稳定」。",
        "reading": "0~1；常与 VMA 配合看：量能波动大说明筹码换手没有规律。",
        "example": "过去 {d} 日成交量在 500 万~2000 万股之间跳动 ⇒ VSTD{d} 通常在 0.4 以上。",
        "caveats": "和 VMA 同源且同属一个「量」的信息块，组合里留一个就够。",
        "refs": [REF_QLIB_DOC],
    },
    "VSUMP": {
        "intuition": "过去 {d} 日「放量之和」占「成交量变化绝对值之和」的比重。",
        "reading": "0~1；高说明近期量能变化主要由放量贡献。",
        "example": "近 {d} 日成交量变化的绝对值合计中，放量部分占 65% ⇒ VSUMP{d} = 0.65。",
        "caveats": "与 VSUMN、VSUMD 同族，三选一即可；这类「变化方向结构」因子的信噪比普遍不高。",
        "refs": [REF_QLIB_DOC],
    },
    "VSUMN": {
        "intuition": "过去 {d} 日「缩量之和」占「成交量变化绝对值之和」的比重。",
        "reading": "0~1；与 VSUMP 互补。",
        "example": "缩量部分占 35% ⇒ VSUMN{d} = 0.35。",
        "caveats": "与 VSUMP 严格互补，同时用只多一份冗余。",
        "refs": [REF_QLIB_DOC],
    },
    "VSUMD": {
        "intuition": "放量之和与缩量之和之差，再除以变化绝对值之和，量的是「量能变化更偏向放量还是缩量」。",
        "reading": "-1~+1；正 = 放量占优。",
        "example": "放量占 65%、缩量占 35% ⇒ VSUMD{d} = +0.30。",
        "caveats": "和 VSUMP/CNTP 同类，注意别把同一类「计数」信息堆在一起。",
        "refs": [REF_QLIB_DOC],
    },
}

# --------------------------------------------------------------------------
# alpha8: eight hand-picked price/volume factors.
# --------------------------------------------------------------------------
FACTOR_PRIMERS: Dict[str, Dict[str, Any]] = {
    "alpha8.alpha1": {
        "intuition": "当天实体涨跌占当日振幅的比例——把「涨了多少」换算成「在今天这条波动区间里占了多少」。",
        "reading": "-1~+1；接近 +1 表示几乎收在最高且当天是涨的。",
        "example": "开 10.00、收 10.40、高 10.45、低 9.95，alpha1 = 0.40/0.50 = 0.80。",
        "caveats": "和 alpha158 的 KMID2 是同一类东西，别指望它提供独立信息。",
        "refs": [REF_QLIB_DOC],
    },
    "alpha8.alpha2": {
        "intuition": "上冲幅度相对下影线长度，量的是「上方被打回的程度」。",
        "reading": "值大 = 冲得很高但收不住；负数 = 杀得很深但被拉回。",
        "example": "前收 10.00、最高 10.50、收 10.10、最 Low 10.05，alpha2 = (10.50-10.00)/(10.10-10.05) = 10.0。",
        "caveats": "分母（当日下影）可以非常小，比值会被放大成噪声，读的时候要看绝对幅度。",
        "refs": [REF_QLIB_DOC],
    },
    "alpha8.alpha3": {
        "intuition": "单位振幅承载了多少成交量——同样的波动，是放量砸出来的还是缩量磨出来的。",
        "reading": "量纲是「手/元」，跨股票不可比，只能在自己的历史里比较。",
        "example": "当日成交 100 万股、振幅 0.50 元，alpha3 = 100万/0.50 = 200 万（股/元）。",
        "caveats": "绝对量纲没有横截面意义，做横截面排序时几乎没有区分度。",
        "refs": [REF_QLIB_DOC],
    },
    "alpha8.alpha4": {
        "intuition": "收盘价在当日区间内的位置（用价格差而不是比例表示）。",
        "reading": "正数 = 收在区间上半部，负数 = 收在下半部。",
        "example": "开 10.00、收 10.30、高 10.40、低 9.90，alpha4 = (10.30-9.90)-(10.40-10.30) = +0.30。",
        "caveats": "价格差没有做归一化，高价股天然占优，做横截面比较前必须中性化或归一。",
        "refs": [REF_QLIB_DOC],
    },
    "alpha8.alpha5": {
        "intuition": "5 日均线与 20 日均线之差，量的是「短期均线跑在长期均线前面多少」。",
        "reading": "正值 = 短期均线在上（多头排列雏形）；数值单位是元，跨股票不可比。",
        "example": "5 日均价 10.20、20 日均价 9.90，alpha5 = +0.30。",
        "caveats": "差值口径受价格水平影响，横截面上其实是在比「价格涨了多少」而不是「均线形态」。",
        "refs": [REF_QLIB_DOC],
    },
    "alpha8.alpha6": {
        "intuition": "近 5 日成交量波动与近 20 日成交量波动之比，量的是「量能的不稳定程度在放大还是收敛」。",
        "reading": "大于 1 = 近期量能比中期更跳（情绪升温）；小于 1 = 量能趋于稳定。",
        "example": "5 日量标准差 800 万股、20 日量标准差 500 万股，alpha6 = 1.6。",
        "caveats": "同一族里已经有一个「量」的因子，与 alpha158 的 VSTD 类高度相关。",
        "refs": [REF_QLIB_DOC],
    },
    "alpha8.alpha7": {
        "intuition": "5 日内最高价相对 5 日前收盘的涨幅，量的是「这 5 天冲得有多猛」。",
        "reading": "10% 表示 5 天里最高冲到过 +10%。",
        "example": "5 天前收 10.00、期间最高 11.00，alpha7 = (11.00-10.00)/10.00 = +10%。",
        "caveats": "它记录的是期间极值而不是当前状态，信号在冲高当天就已经兑现了，"
                   "用它做次日信号要特别小心时间对齐。",
        "refs": [REF_QLIB_DOC],
    },
    "alpha8.alpha8": {
        "intuition": "3 日涨幅减 10 日涨幅，量的是「近期走势相对中期是加速还是减速」。",
        "reading": "正数 = 最近 3 天涨得比 10 天平均更快（动量加速）。",
        "example": "3 日涨幅 +6%、10 日涨幅 +10%，alpha8 = -4%（近期在减速）。",
        "caveats": "两个不同窗口的涨跌幅相减，量纲上是「百分点差」，"
                   "短窗口噪声大，10 日窗口在震荡市里几乎没有方向性。",
        "refs": [REF_QLIB_DOC],
    },

    # ---- fm1：基本面与估值面板 ----
    "fm1.rev_yoy": {
        "intuition": "最近一期 TTM 营收的同比增速，量的是「生意有没有变大」。",
        "reading": "正值代表增长；A 股中位数大致在 5%~15% 区间，超过 50% 要看是不是并表或低基数。",
        "example": "TTM 营收 12.0 亿，去年同期 10.0 亿，rev_yoy = +20%。",
        "caveats": "低基数会让同比失真；口径取滞后一期（lag=1）是为了避免用到未披露的报告期。"
                   "增速高不等于股价会涨，高增长往往已经被定价。",
        "refs": [REF_TUSHARE, REF_FF5],
    },
    "fm1.rev_cagr": {
        "intuition": "把营收的历史序列压缩成一个数——整个可用区间内的年均复合增长率。",
        "reading": "代表「长期成长速度」，比单年同比更稳定；典型 0~30%。",
        "example": "3 年前营收 6.0 亿、现在 10.0 亿，rev_cagr = (10/6)^(1/3)-1 ≈ 18.6%。",
        "caveats": "可用历史长度不同的股票之间不可比（新股只有一两年，CAGR 会非常不稳定）；"
                   "并购或重组会打断序列。",
        "refs": [REF_TUSHARE],
    },
    "fm1.rev_yoy_std": {
        "intuition": "营收同比增速序列的标准差，量的是「增长的稳定程度」。",
        "reading": "越小越稳；A 股稳健型公司常见 5%~15%。",
        "example": "过去 4 期同比为 18%、15%、22%、17%，标准差约 2.8%。",
        "caveats": "低于两期同比时为 null（样本不足）；"
                   "「稳定但停滞」的低波动公司容易被误当成高质量成长。",
        "refs": [REF_TUSHARE],
    },
    "fm1.profit_yoy": {
        "intuition": "归母净利润（扣非、滞后一期）的同比增速，量的是「盈利扩张还是收缩」。",
        "reading": "正负皆可读；利润增速长期高于营收增速说明经营杠杆在起作用。",
        "example": "扣非归母净利 1.20 亿，去年同期 0.80 亿，profit_yoy = +50%。",
        "caveats": "扣非口径会剔除一次性损益，但扣非也可能是负的；"
                   "亏损转盈时同比百分比失去意义，要直接看金额。",
        "refs": [REF_TUSHARE],
    },
    "fm1.profit_positive_ratio": {
        "intuition": "历史上盈利为正的期数占比，量的是「这家公司赚不赚钱」的稳定性。",
        "reading": "1.0 表示每个报告期都赚钱；周期性行业通常在 0.5~0.8 之间波动。",
        "example": "过去 8 个报告期有 6 期盈利为正，profit_positive_ratio = 0.75。",
        "caveats": "样本期数量不同的股票之间不可比；"
                   "「赚钱的比例高」不等于「赚得多」，还要看 ROE 和增长。",
        "refs": [REF_TUSHARE, REF_FF5],
    },
    "fm1.roe": {
        "intuition": "TTM 归母净利除以平均净资产，量的是股东每一元钱的账面资产一年赚回多少。",
        "reading": "A 股常见 5%~20%；ROE 高来自高利润率或高周转，要用杜邦拆开看才知原因。",
        "example": "TTM 净利 2.0 亿、平均净资产 20 亿，roe = 10%。",
        "caveats": "平均净资产用期初期末平均，单期波动大；"
                   "高 ROE 可能来自高杠杆（金融地产常见），也可能来自轻资产高利润率，两者的风险完全不同。",
        "refs": [REF_TUSHARE, REF_FF93],
    },
    "fm1.roe_trend": {
        "intuition": "ROE 序列的斜率，量的是「盈利能力在变好还是变差」。",
        "reading": "正 = 逐期走高；单位是「每期 ROE 变化多少个百分点」。",
        "example": "ROE 从 8% 逐期升到 12%（4 期），roe_trend 约 +1.3 个百分点/期。",
        "caveats": "斜率对序列长度和噪声敏感，期数少时几乎不可用；"
                   "周期股在景气顶部 ROE 斜率最大，恰是风险最大的时候。",
        "refs": [REF_TUSHARE],
    },
    "fm1.rd_ratio": {
        "intuition": "研发费用占营业收入的比例，量的是「这家公司为下一期投入了多少」。",
        "reading": "制造业常见 3%~10%，芯片/医药 15% 以上；未披露时为 null。",
        "example": "研发费用 1.5 亿、营收 20 亿，rd_ratio = 7.5%。",
        "caveats": "研发资本化的公司会把费用资本化，这个口径会低估真实投入，跨公司比较要小心；"
                   "高研发短期压利润，它的收益通常滞后 2~3 年。",
        "refs": [REF_TUSHARE, REF_QFACTOR],
    },
    "fm1.pe_ttm": {
        "intuition": "滚动市盈率：总市值除以最近四个季度净利润，量的是「为一块钱当期利润付多少钱」。",
        "reading": "越低越便宜（方向是「越小越好」）；A 股中位数大致 20~30 倍，行业差异极大。",
        "example": "市值 100 亿、TTM 净利 5 亿，pe_ttm = 20 倍。",
        "caveats": "周期股在盈利高点 PE 反而最低（分母虚高），这是价值陷阱最常见的形态；"
                   "亏损公司 PE 无意义。",
        "refs": [REF_TUSHARE, REF_FF93],
    },
    "fm1.pb": {
        "intuition": "市净率：总市值除以净资产，量的是「为一块钱账面净资产付多少钱」。",
        "reading": "越低越便宜；银行 0.5~0.8、软件 3~8 很正常，跨行业比较几乎没有意义。",
        "example": "市值 80 亿、净资产 40 亿，pb = 2.0 倍。",
        "caveats": "账面净资产会随重估、减值、商誉变动而失真；"
                   "轻资产公司的 PB 天然高，重资产公司天然低。",
        "refs": [REF_TUSHARE, REF_FF93],
    },
    "fm1.momentum_20d": {
        "intuition": "过去 20 个交易日的前复权涨幅，量的是「最近一个月的趋势有多强」。",
        "reading": "正值表示上涨；A 股 20 日动量的横截面差异通常在 ±20% 以内。",
        "example": "20 个交易日前收 10.00、现在 11.20，momentum_20d = +12%。",
        "caveats": "复权口径必须前后一致（用前复权或后复权都可以，但不能混）；"
                   "停牌期间价格不变会把动量稀释，动量和反转都可能因此失真。",
        "refs": [REF_MOMENTUM],
        "detail": [
            {"heading": "它想抓什么", "body":
             "标准动量假设：过去涨得好的股票会继续涨。窗口选 20 个交易日，"
             "对应一个月的调仓周期，也是系统 20 日 label 的天然搭档。"},
            {"heading": "为什么是这个口径", "body":
             "用复权收盘价比值而不是对数收益，是为了和 `volume_price` 表的复权口径"
             "一一对应；停牌日不产生新价格，所以停牌越多的股票动量越被稀释。"},
            {"heading": "系统里承担什么角色", "body":
             "它是 raw5 五个特征之一，被 live 模型 `research-csi1000-clean-raw5` 使用。"
             "它在这一族里不是最强的，但它和 `momentum_60d` 分别覆盖 1 个月和 3 个月两个"
             "尺度，模型可以自己学两者的差（加速/减速）。"},
            {"heading": "实测结论怎么读", "body":
             "csi1000、145 个 20 日非重叠截面、next-open、未扣成本：**最优方向是「低」**，"
             "h20 IC +0.0526（t 4.43），h60 +0.0503（t 4.30）——在 A 股小盘上这是"
             "**反转**而不是动量。另一个口径（2020–2024、hs300/csi500）单因子 IC −0.0352"
             "（t −2.10）不显著，但它所在的 5 因子组让组合 IC 近乎翻倍："
             "**单因子不显著不等于它在组合里没有位置**。"},
            {"heading": "常见误用", "body":
             "① `finance.factor_definitions` 里 `direction` 写的是 `higher_is_better`，"
             "那是教科书先验、不是实测结论；按实测方向用（越低越好），符号拍反会让 IC 减半"
             "（`raw5_eq` 0.060 → `raw5_measured` 0.113）。"
             "② 与 `vault.return_21d` 只差一个交易日，两个一起用等于重复计权。"},
        ],
    },
    "fm1.momentum_60d": {
        "intuition": "过去 60 个交易日的涨幅，量的是「一个季度的趋势」。",
        "reading": "窗口越长越稳，但反应越慢；典型区间 ±40%。",
        "example": "60 个交易日前收 10.00、现在 9.00，momentum_60d = -10%。",
        "caveats": "60 日窗口里有一次分红除权就会错一个台阶，复权必须正确；"
                   "半年以上涨幅在 A 股常常已经反转（1 年期反转效应）。",
        "refs": [REF_MOMENTUM],
        "detail": [
            {"heading": "它想抓什么", "body":
             "和 `momentum_20d` 同一个假设，只是窗口拉到 3 个月。窗口越长，"
             "单日噪声占比越低，信号越稳，但对趋势拐点越迟钝。"},
            {"heading": "为什么是这个口径", "body":
             "60 个交易日约等于一个季度，正好跨过一份财报的披露周期；"
             "它和 20 日并存的意义不是「两个动量」，而是让模型可以观察两者的差，"
             "也就是「近一个月相对过去三个月是加速还是减速」。"},
            {"heading": "系统里承担什么角色", "body":
             "raw5 五个特征之一，被 live 模型 `research-csi1000-clean-raw5` 使用。"
             "注意它在注册表里**没有任何单因子 test 行**：它的地位完全来自"
             "组合层面的表现（raw5 的联合 IC），不是单因子检验挣来的。"},
            {"heading": "实测结论怎么读", "body":
             "csi1000、145 个 20 日非重叠截面：**最优方向同样是「低」**（反转），"
             "h20 IC +0.0647（t 4.65）、h60 +0.0561（t 4.74），比 20 日版更强。"
             "也就是说在 csi1000 上，**过去三个月跌得多的股票后续更好**，"
             "这与「成长动量」的选股偏好方向相反。"},
            {"heading": "常见误用", "body":
             "① 把 `direction=higher_is_better` 直接当成使用方向——实测是反的。"
             "② 60 日窗口内的除权如果没对齐，会给整个序列加一个假台阶，"
             "而这个台阶恰好和「除权日集中度」相关，容易伪造成一个看起来显著的信号。"
             "③ 与 `momentum_20d` 相关性高，横截面 rank 后两者常常只差几档，"
             "同时入模时要看共线性而不是只看各自 IC。"},
        ],
    },
    "fm1.volatility_20d": {
        "intuition": "过去 20 日日收益率标准差乘以 √252 年化，量的是「这只股票平时有多颠」。",
        "reading": "A 股个股常见 30%~60%；年化后是百分比，直接可比。",
        "example": "日收益标准差 3% → 年化 3% × √252 ≈ 47.6%。",
        "caveats": "极端日（跌停、停牌复牌）会主导整个标准差；"
                   "低波动组合长期有效，但在成长风格行情里会被明显压制。",
        "refs": [REF_ARCH],
        "detail": [
            {"heading": "它想抓什么", "body":
             "低波动异象：波动率高的股票长期风险调整后收益更差。"
             "在 A 股小盘股上这条尤其强——高波动常常等于高投机、高换手、高成本。"},
            {"heading": "为什么是这个口径", "body":
             "用日收益标准差而不是振幅（high−low），是因为标准差对跳空和"
             "涨跌停的刻画更完整；乘 √252 只是把它换算成年化百分比，"
             "**对横截面排序没有任何影响**（单调变换），所以换不年化不改变模型信号，"
             "只改变你读数的习惯。"},
            {"heading": "系统里承担什么角色", "body":
             "raw5 五个特征之一，被 live 模型使用。它同时是「风险」这一层的代表："
             "模型里其他四个特征在描述方向，而它在描述不确定性。"},
            {"heading": "实测结论怎么读", "body":
             "csi1000、145 个 20 日非重叠截面：低波动最优，h20 IC +0.0950（t 6.12）、"
             "h60 +0.1188（t 8.05），是 raw5 里第二强的单因子。"
             "更关键的一条：候选因子 `amplitude_10d` 与它相关 **0.864**，"
             "而 amplitude 的增量检验 MDE₈₀ 只有 0.0021 —— 这是一个**强阴性**："
             "有把握说「振幅类因子的增量 < 0.0021」，因为这块信息已经被波动率吃掉了。"},
            {"heading": "常见误用", "body":
             "① 把「低波动好」当成「低波动股票一定涨」——它赚的是横截面差异，"
             "整体熊市里低波动股照样跌。② 用未复权价格算日收益：除权日会出现一个"
             "假暴跌，直接被记成高波动。③ 停牌复牌后的第一根 K 线收益极端，"
             "样本少时会把整段标准差带偏。"},
        ],
    },
    "fm1.turnover_20d": {
        "intuition": "过去 20 日日换手率的均值，量的是「筹码换手的活跃度」，属于情绪/流动性因子。",
        "reading": "A 股主板常见 1%~5%，题材股 10%+；换手越高说明交易越拥挤。",
        "example": "过去 20 日日均换手 3.2%，turnover_20d = 3.2%。",
        "caveats": "高换手意味着高交易成本和更高的冲击成本，"
                   "任何依赖它的收益都必须先扣成本；T+1 下当日买进的筹码不能当日卖出。",
        "refs": [REF_ILLIQ, REF_SSE],
        "detail": [
            {"heading": "它想抓什么", "body":
             "换手率衡量的是「有多少筹码在换手」。在 A 股它同时混了两件事："
             "流动性（容易买卖）和拥挤度（太多人在交易同一只票）。"
             "低换手溢价主要来自后者：被冷落的股票一旦有资金进入，弹性更大。"},
            {"heading": "为什么是这个口径", "body":
             "取日均而不是最新一天，是为了压掉单日的放量噪声（涨停日的换手可以"
             "是平时的十倍）。数据取自 `financial_daily_basic.turnover_rate`，"
             "不是自己用成交量除以流通股本算的——这个口径和 Tushare 一致，"
             "跨源比对时不会差一个小数点。"},
            {"heading": "系统里承担什么角色", "body":
             "它是 raw5 之一（被 live 模型使用），同时以 `candidate` 状态留在注册表里"
             "等更细的净收益检验。这是有意的：**IC 强不等于净收益强**，"
             "高换手因子的成本侵蚀是所有因子里最大的。"},
            {"heading": "实测结论怎么读", "body":
             "csi1000、145 个 20 日非重叠截面：**低换手最优**，"
             "h20 IC +0.1012（t 6.98）、h60 +0.1291（t 9.64），是 raw5 里最强的单因子。"
             "另一口径（2020–2024、hs300/csi500）也一致：hs300 −0.0875（t −3.01）、"
             "csi500 −0.0879（t −3.55），与现有评分的最大相关只有 0.36，"
             "说明它提供的是评分看不见的信息。"},
            {"heading": "常见误用", "body":
             "① 注册表 `direction` 写 `higher_is_better`，实测是**低换手更好**，"
             "按实测方向用。② 直接拿 IC 当收益：低换手组合换手率低，成本也低，"
             "但反过来「高换手因子在减仓时卖出成本更高」，两端都要算。"
             "③ 停牌股的换手率会掉到接近 0，被算成「极低换手」，"
             "在停牌前买入的样本上会伪造成有效信号。"},
        ],
    },
    "fm1.log_market_cap": {
        "intuition": "流通市值的自然对数，量的是「公司有多大」。",
        "reading": "方向中性——不是越大越好或越小越好，它只是把市值压缩到一个可比的尺度上。",
        "example": "流通市值 50 亿 ⇒ log(50 亿 × 1e4) ≈ 26.9（自然对数）。",
        "caveats": "A 股小市值效应极强，很多因子不做市值中性化就只是在抓市值暴露；"
                   "中位数、正态化程度都要检查。",
        "refs": [REF_TUSHARE],
        "detail": [
            {"heading": "它想抓什么", "body":
             "规模因子。取对数不是为了「好看」，而是因为市值分布严重右偏："
             "直接拿市值做横截面排序，一只巨无霸就能把整个分布压扁；"
             "取对数后中位数和均值回到同一量级。"},
            {"heading": "为什么是这个口径", "body":
             "用**流通**市值（`circ_mv`，单位万元）而不是总市值：A 股大量公司的"
             "限售股占比很高，总市值会把「实际能交易的规模」放大好几倍。"
             "`circ_mv <= 0` 记为 NaN，不用 0 填充——0 取对数会变成负无穷，"
             "而且 0 本身就不该出现在真实市值里。"},
            {"heading": "系统里承担什么角色", "body":
             "raw5 五个特征之一。它在模型里还有第二重身份：**风格暴露的刻度尺**。"
             "任何声称有 alpha 的因子，都要先看它和 log_market_cap 的相关性，"
             "否则可能只是在抓小市值。"},
            {"heading": "实测结论怎么读", "body":
             "csi1000、145 个 20 日非重叠截面：**小市值更好**，"
             "h20 IC +0.0402（t 3.36）、h60 +0.0462（t 3.55）。"
             "这是 raw5 里最弱的单因子，但也是最「老」的异象，"
             "在 A 股小盘池里几乎是所有其他因子的共同底色。"
             "另有一处必须知道的历史缺陷：早期 panel 里这一列 **100% 是 NaN**，"
             "原因是建 panel 的顺序错了（label 先挂载、因子后挂载），"
             "不是数据缺失——`quant_data.financial_daily_basic.circ_mv` 一直是可用的。"},
            {"heading": "常见误用", "body":
             "① 用 `fillna(0)` 补缺失市值：0 标准化后约等于 −13 个标准差，"
             "是极端杠杆点，会同时污染 scaler 和系数（这是真实发生过的 bug）。"
             "② 把「小市值有效」直接做成「只买最小的」：小市值股的容量极低，"
             "冲击成本会把超额吃光，而且退市新规之后这个效应的强度在变化。"
             "③ 用总市值和流通市值混着比：两个口径能差一倍以上。"},
        ],
    },

    # ---- vault 手写因子 ----
    "vault.bias_turn_21d_252d": {
        "intuition": "近期（21 日）换手率相对其自身长期水平（252 日）的偏离，量的是「最近资金参与度是否突然放大」。",
        "reading": "正数 = 近期比长期活跃；突然放大的换手常伴随消息面。",
        "example": "21 日均换手 4%、252 日均换手 2%，偏离度 +100%。",
        "caveats": "252 日窗口需要一年多的历史，新股样本不足；"
                   "高换手拥挤的股票在放量之后往往先跌后稳，方向并不确定。",
        "refs": [REF_ILLIQ],
        "detail": [
            {"heading": "它想抓什么", "body":
             "和 `turnover_20d` 同族，但问的是不同的问题。`turnover_20d` 是**横截面**比较"
             "「谁比谁换手高」；这个是**时序**比较「这只票现在比它自己平时热闹多少」。"
             "前者抓「一直很冷的股票」，后者抓「突然变热的股票」。"},
            {"heading": "为什么是这个口径", "body":
             "用 21 日对 252 日，是让短窗口恰好等于一个调仓月、长窗口恰好等于一年，"
             "两个窗口的比值天然无量纲，跨股票可比，不需要再做行业或市值中性化。"
             "这也是它相对 `turnover_20d` 的唯一优势：绝对值换手率在大小盘之间差一个量级，"
             "而「相对自己的偏离」不受这个影响。"},
            {"heading": "系统里承担什么角色", "body":
             "注册表状态 `candidate`，**没有接入任何模型**。它现在的价值是作为一个"
             "「候选的拥挤度信号」占位：如果以后要做「避开刚被爆炒的股票」"
             "这类风控，它是现成的候选。"},
            {"heading": "实测结论怎么读", "body":
             "2020–2024、h20、next-open、未扣成本：hs300 IC −0.0400（t −2.85）、"
             "csi500 −0.0500（t −2.69），与现有评分的最大相关只有 0.35，"
             "两个池方向一致且都过 t 检验，所以给 `candidate` 而不是 `confirmed`"
             "——还缺净收益和 OOS。"},
            {"heading": "常见误用", "body":
             "① 和 `turnover_20d` 一起放进模型：两者同族，横截面相关不低，"
             "同时入模会让「换手」这一族挤占过多权重。"
             "② 新股上市不满 252 个交易日时长期均换手不可用，"
             "这个因子的分母会极不稳定，必须设最小样本期。"
             "③ 把它当成「动量确认」：换手放大既可能是资金进入，也可能是"
             "出货，方向要结合价格结构判断，不能单看换手。"},
        ],
    },
    "vault.return_21d": {
        "intuition": "21 个交易日的简单涨幅，和 fm1.momentum_20d 是同一类东西。",
        "reading": "正值上涨，典型 ±25%。",
        "example": "21 个交易日前 10.00、现在 11.50，return_21d = +15%。",
        "caveats": "与 fm1.momentum_20d 高度重复（只差一个交易日的窗口）；"
                   "两者在同一组合里会互相稀释权重。",
        "refs": [REF_MOMENTUM],
    },
    "vault.beta_60d_000300": {
        "intuition": "过去 60 日个股相对沪深 300 的回归斜率，量的是「这只股票跟着大盘走得多紧」。",
        "reading": "1 = 与沪深 300 同涨同跌；大于 1 是高 beta（放大波动），小于 1 是防御型。",
        "example": "沪深 300 涨 10% 时个股平均涨 15%，beta = 1.5。",
        "caveats": "beta 是相对某个指数定义的，换基准数值就变；"
                   "beta 高在熊市里意味着跌得更多，这不是「更好」而是「波动更大」。",
        "refs": [REF_CAPM, REF_CSINDEX],
    },
    "vault.gpm_ttm": {
        "intuition": "TTM 毛利率，量的是「每一元收入里留下多少毛利」，是产品定价权的直接体现。",
        "reading": "消费 40%~60%、制造业 10%~30%、周期 5%~20%；长期稳定上行是好信号。",
        "example": "TTM 营收 20 亿、毛利 8 亿，gpm_ttm = 40%。",
        "caveats": "毛利率是账面口径，会被会计政策变更和成本结转方式影响；"
                   "行业之间不可比，只能在行业内部做横截面比较。",
        "refs": [REF_GROSSPROFIT, REF_TUSHARE],
    },
    "vault.yoy_roe": {
        "intuition": "ROE 的同比变化，量的是「盈利能力相对去年是改善还是恶化」。",
        "reading": "正数 = 改善；典型 ±5 个百分点。",
        "example": "去年 ROE 8%、最新 11%，yoy_roe = +3 个百分点。",
        "caveats": "ROE 同比混入了分母（净资产）的变化，不等于利润在增长；"
                   "与 fm1.roe_trend（逐期斜率）高度相关，两者只需留一个。",
        "refs": [REF_FF93],
    },
    "vault.q_ia": {
        "intuition": "q 因子框架里的「投资 / 资产」，量的是「企业往里投了多少钱」，高投资通常对应低未来收益。",
        "reading": "相对同行业偏高 = 激进扩张；A 股扩张期里高投资往往是景气高点。",
        "example": "净投资 5 亿、总资产 100 亿，I/A = 5%。",
        "caveats": "总资产口径对商誉和并购敏感；"
                   "这个因子在 A 股的表现依赖 regime，成长牛里往往是反着的。",
        "refs": [REF_QFACTOR],
    },
    "vault.q_roe": {
        "intuition": "q 因子框架里的「盈利能力」，量的是「每一元资产产生的经营利润」，比 ROE 更干净。",
        "reading": "长期高位的公司才具备持续性；单年跳升通常来自周期或一次性项目。",
        "example": "经营利润 8 亿、总资产 100 亿，ROE(q) = 8%。",
        "caveats": "需要利润总额中剔除利息与税收，口径和财报原始行项目对不齐时容易算错；"
                   "与 fm1.roe 相关但不是同一个东西（分母是总资产而非净资产）。",
        "refs": [REF_QFACTOR],
    },
    "vault.beta_252": {
        "intuition": "过去 252 个交易日（一年）相对沪深 300 的回归斜率，量的是「一年的风险敞口」。",
        "reading": "典型 0.5~1.8；值大 = 系统性风险敞口高。",
        "example": "一年里沪深 300 涨 20%、个股平均涨 30%，beta = 1.5。",
        "caveats": "窗口长、样本足，但结构变化（行业切换、重大重组）会让整段 beta 失真；"
                   "与 60 日 beta 常常给出不同的画像，不是谁对谁错。",
        "refs": [REF_CAPM],
    },
    "vault.analyst_fy1": {
        "intuition": "分析师对本财年的盈利一致预期，量的是「市场共识认为公司会赚多少」。",
        "reading": "绝对值随行业差异极大，要看「预期变化」和「预期 vs 实际」才有意义。",
        "example": "一致预期全年净利 8 亿。",
        "caveats": "覆盖度低的股票「一致预期」其实是一两个人的观点；"
                   "财报前预期会被反复调整，事件窗口里方向常常相反。",
        "refs": [REF_TUSHARE],
    },
    "vault.analyst_fy2": {
        "intuition": "分析师对下一财年的盈利预期，量的是「共识对公司更远期的看法」。",
        "reading": "FY2/FY1 的比值隐含了市场预期的年增速。",
        "example": "FY1 预期 8 亿、FY2 预期 10 亿，隐含增速 25%。",
        "caveats": "远期预测的不确定性远大于当期；"
                   "分析师覆盖少的时候，两个 FY 的差异可能只是模型外推而不是真实共识。",
        "refs": [REF_TUSHARE],
    },
    # ---- alpha101（本仓库没有公式实现，只登记了编号）----
    "alpha101.1": {
        "intuition": "WorldQuant《101 Formulaic Alphas》里的第 1 个，属于「量价背离 + 秩变换」那一族，"
                     "设计目的是把短期噪声滤掉后保留排序信息。",
        "reading": "Alpha101 的输出天然是截面排序值，使用方式是多空分组而不是看单只股票。",
        "example": "表达式见官方论文与 Qlib 实现；本仓库只登记了编号，没有公式实现。",
        "caveats": "Alpha101 的原始持有期设计只有 0.6~6.4 天，"
                   "放进 20 日口径的模型属于「持有期错配」——测不出来不等于因子无效。"
                   "另外论文只给数学表达式不给业务注释，各家实现的算子语义可能不同。",
        "refs": [REF_ALPHA101_PAPER, REF_QLIB_SRC],
    },
    "alpha101.12": {
        "intuition": "Alpha101 第 12 个，属于「日内位置 × 成交量」的组合，量的是资金在当日区间里的进出偏好。",
        "reading": "同样以截面排序使用；单票读数没有意义。",
        "example": "表达式见官方论文与 Qlib 实现；本仓库只登记了编号。",
        "caveats": "与 alpha101.1 同样受持有期错配影响；"
                   "同一族里多个公式高度相关，同时放入会挤占彼此的权重。",
        "refs": [REF_ALPHA101_PAPER, REF_QLIB_SRC],
    },
    "alpha101.33": {
        "intuition": "Alpha101 第 33 个，是「低点距离 × 成交额」的量价组合，量的是低位承接的力度。",
        "reading": "同样是排序型因子；它与 alpha101.101 在本仓库的历史检验里相关 −0.93，是同一现象的正反两面。",
        "example": "表达式见官方论文与 Qlib 实现；本仓库只登记了编号。",
        "caveats": "与 alpha101.101 近乎互为反号，选一个就够；"
                   "短持有期设计在 20 日口径里会显著衰减。",
        "refs": [REF_ALPHA101_PAPER, REF_QLIB_SRC],
    },
    "alpha101.101": {
        "intuition": "Alpha101 第 101 个，属于「日内相对位置」这一族，是 101 个里被讨论最多的一个。",
        "reading": "排序型因子；和 alpha101.33 相关 −0.93，两者方向相反。",
        "example": "表达式见官方论文与 Qlib 实现；本仓库只登记了编号。",
        "caveats": "公开即衰减的现象在 Alpha101 上尤其明显；"
                   "1 日口径有信号、20 日口径测不出，多半是持有期错配而不是因子无效。",
        "refs": [REF_ALPHA101_PAPER, REF_QLIB_SRC],
    },
    # ---- 组合因子 ----
    "group.value": {
        "intuition": "价值组的组合因子：把 pe_ttm 与 pb 合成一个「便宜程度」的排序。",
        "reading": "值越大越便宜（与成员因子方向相反，因为成员是越小越好）。",
        "example": "pe_ttm 15 倍、pb 1.2 倍，在全市场属于偏低分位。",
        "caveats": "价值陷阱是 A 股最常见的因子失效形态：便宜是因为基本面在恶化。"
                   "组合因子会掩盖成员之间的分歧，排查时必须回到成员因子逐个看。",
        "refs": [REF_FF93],
    },
    "group.quality": {
        "intuition": "质量组：把 ROE 与毛利率合成一个「赚钱能力」的排序。",
        "reading": "值越大质量越好。",
        "example": "roe 15%、gpm_ttm 45% 属于高质量象限。",
        "caveats": "两成员都依赖财报，且高度同源（高 ROE 常常来自高毛利率）；"
                   "在周期股上「质量」会变成「景气高点」的代理变量。",
        "refs": [REF_FF5, REF_GROSSPROFIT],
    },
    "group.growth": {
        "intuition": "成长组：把营收同比与利润同比合成一个「成长速度」的排序。",
        "reading": "值越大增长越快。",
        "example": "rev_yoy +25%、profit_yoy +40% 属于高成长。",
        "caveats": "高成长已被市场定价的比率很高；"
                   "同时 A 股小盘股天然高成长，不中性化市值就只是在抓小市值暴露。",
        "refs": [REF_FF5],
    },
    "group.value_quality_growth": {
        "intuition": "把价值、质量、成长三组合成一个多因子排序，属于经典的「Fama-French 三要素」思路。",
        "reading": "综合分越高代表「便宜 + 能赚钱 + 在长大」同时成立。",
        "example": "一个 PE 12 倍、ROE 18%、营收增速 30% 的公司在三组里都排前面。",
        "caveats": "三组内部相关性不低，等权合成容易变成某一个风格因子的影子；"
                   "合成后仍要做增量检验（ΔIC），不能只看绝对 IC。",
        "refs": [REF_FF5, REF_MULTIPLE_TESTING],
    },
    "group.course_44": {
        "intuition": "课程第 44 讲的量价因子组合，是外部课程里的一套多因子打包方案。",
        "reading": "作为整体排序使用。",
        "example": "具体构成见课程对应的因子实现（jupiter_script）。",
        "caveats": "外部课程因子的常见问题是：口径未公开、样本期选择性报告、"
                   "无法复现收益曲线。引用前必须做独立复算，不要直接采信其历史结论。",
        "refs": [REF_QLIB_DOC],
    },
}

# --------------------------------------------------------------------------
# Family fallbacks: used when a factor has no primer of its own. The
# registry document's ``family`` field decides which one applies.
# --------------------------------------------------------------------------
FAMILY_PRIMERS: Dict[str, Dict[str, Any]] = {
    "K线形态": {
        "intuition": "日内 K 线形状类因子：只描述当天这一根 K 线里多空各自占了多少，不含时间序列信息。",
        "reading": "取值通常在 -1 到 +1 之间，正负表示多空方向。",
        "example": "一根长阳线的实体占比会接近 +1。",
        "caveats": "单日信息对次日预测力很弱；这一族内部冗余度极高，"
                   "alpha158 里 KMID/KMID2/KSFT/KSFT2 几乎是同一件事。",
        "refs": [REF_QLIB_DOC],
    },
    "价格位置": {
        "intuition": "用开盘/最高/最低/VWAP 相对收盘价的位置，刻画当天价格走过的路径。",
        "reading": "都是无量纲比值，1 附近表示「和收盘价差不多」。",
        "example": "HIGH = 1.03 表示当天最高价比收盘高 3%。",
        "caveats": "是「幅度」不是「方向」；和 K 线形态族信息高度重叠。",
        "refs": [REF_QLIB_DOC],
    },
    "趋势": {
        "intuition": "把最近一段时间的价格拟合成一条趋势，量的是斜率、拟合优度和偏离程度。",
        "reading": "斜率看方向，R² 看干净程度，残差看短期偏离。",
        "example": "20 日斜率为正、R² 0.8、残差 +5% —— 趋势明确，但今天已经偏高。",
        "caveats": "这一族在单边行情里有效、在震荡行情里失效，跨 regime 迁移能力弱。",
        "refs": [REF_MOMENTUM],
    },
    "波动": {
        "intuition": "度量价格波动幅度的一族因子。",
        "reading": "值越大越颠；低波动组合长期有效但在成长行情里会被压制。",
        "example": "STD20 = 4% 表示过去 20 日的收盘价标准差是收盘价的 4%。",
        "caveats": "极端行情日会主导整个统计量；多个波动率因子同时入模会挤占同一份风险信息。",
        "refs": [REF_ARCH],
    },
    "区间位置": {
        "intuition": "当前价格在最近 N 日高低区间里处于什么位置（分位、前高前低距离、极值出现的时间）。",
        "reading": "分位数 0~1 最好读，其余需要和对应区间的极值一起看。",
        "example": "RANK20 = 0.95 表示当前价在 20 日里排倒数第 1。",
        "caveats": "位置高不等于贵：强趋势里位置高是常态；"
                   "「高位就该跌」是反转型叙事，在趋势行情里会持续失效。",
        "refs": [REF_QLIB_DOC],
    },
    "量价相关": {
        "intuition": "衡量价格和成交量之间是否同步的因子（相关系数）。",
        "reading": "-1 到 +1；正 = 同向，负 = 背离。",
        "example": "放量上涨、缩量下跌通常给出 +0.4 以上的 CORR。",
        "caveats": "A 股散户主导，量价关系经常阶段性反转；相关不等于因果。",
        "refs": [REF_ILLIQ],
    },
    "涨跌结构": {
        "intuition": "用上涨/下跌的「次数占比」或「幅度占比」刻画一段时间里涨跌的构成。",
        "reading": "0~1 或 -1~+1，取决于具体算子。",
        "example": "20 日里 14 天涨 6 天跌，CNTP = 0.70。",
        "caveats": "只数次数不看幅度会丢掉最重要的信息；这一族内部几乎全是互补或重复的变体。",
        "refs": [REF_QLIB_DOC],
    },
    "成交量": {
        "intuition": "以成交量为核心的一族因子，描述放缩量结构与量能稳定度。",
        "reading": "多为无量纲比值，1 附近表示「和近期平均水平一致」。",
        "example": "VMA20 = 0.67 表示今日成交量是 20 日均量的 1.5 倍。",
        "caveats": "量能结构常与「拥挤度/情绪」高度重合；高换手意味着高成本，收益必须先扣成本。",
        "refs": [REF_ILLIQ],
    },
    "panel": {
        "intuition": "基本面面板因子：来自财报（营收、利润、ROE、研发）与估值、行情字段的标准化指标。",
        "reading": "多数以「越大越好 / 越小越好 / 中性」三种方向之一使用。",
        "example": "roe = 10% 意味着平均每 10 元净资产一年赚 1 元。",
        "caveats": "财报口径必须点时（ann_date <= score_date）；"
                   "没有点时约束的基本面回测几乎必然含有未来函数。",
        "refs": [REF_TUSHARE],
    },
    "liquidity": {
        "intuition": "流动性与换手类因子，量的是「筹码转手的速度」。",
        "reading": "换手率、偏离度一类的相对量。",
        "example": "21 日均换手 4% 相对 252 日均换手 2%，偏离 +100%。",
        "caveats": "换手率越高，交易成本和冲击成本越高，"
                   "任何基于它的收益都必须先做成本调整再谈有效性。",
        "refs": [REF_ILLIQ, REF_SSE],
    },
    "momentum": {
        "intuition": "动量类因子：最近一段时间的累计涨幅（或其倒数形式）。",
        "reading": "正数代表上涨（注意有些实现取过倒数）。",
        "example": "20 日涨幅 +12%。",
        "caveats": "窗口越长信号越稳但换手越低；"
                   "复权口径不一致会让整个序列错一个台阶。",
        "refs": [REF_MOMENTUM],
    },
    "quality": {
        "intuition": "质量类因子：盈利能力、利润率、盈利稳定性。",
        "reading": "越大通常越好。",
        "example": "gpm_ttm = 40% 表示每一元收入留下四毛毛利。",
        "caveats": "会计政策、一次性损益和杠杆会同时污染口径；"
                   "周期股在景气顶部的「高质量」恰恰是风险信号。",
        "refs": [REF_FF5, REF_GROSSPROFIT],
    },
    "growth": {
        "intuition": "成长类因子：营收与利润的同比增速、CAGR。",
        "reading": "越大越快。",
        "example": "TTM 营收 12 亿 vs 去年 10 亿，rev_yoy = +20%。",
        "caveats": "低基数会让增速失真；A 股高成长与高估值、高换手高度相关。",
        "refs": [REF_TUSHARE],
    },
    "valuation": {
        "intuition": "估值类因子：PE、PB 等价格相对基本面的比值。",
        "reading": "越小越便宜（注册表里方向标为「越小越好」）。",
        "example": "PE 15 倍、PB 1.2 倍在各自行业内属于偏低分位。",
        "caveats": "周期股盈利高点时 PE 最低，是价值陷阱的典型形态；"
                   "跨行业比较必须先按行业中性化。",
        "refs": [REF_FF93],
    },
    "risk": {
        "intuition": "风险类因子：beta、波动率一类的系统性风险敞口度量。",
        "reading": "值越大表示越贴近大盘、波动越大。",
        "example": "beta_252 = 1.5 表示沪深 300 涨 10% 时该股平均涨 15%。",
        "caveats": "beta 依赖所选基准指数，换基准数值就变；"
                   "低风险因子在成长行情里会长期跑输。",
        "refs": [REF_CAPM],
    },
    "analyst": {
        "intuition": "分析师预期类因子：卖方一致预期及其变化。",
        "reading": "绝对水平跨行业不可比，要看预期调整方向和预期与实际的差异。",
        "example": "FY1 一致预期净利 8 亿。",
        "caveats": "覆盖度低的股票几乎没有真正的「共识」；事件窗口里预期因子的方向常常相反。",
        "refs": [REF_TUSHARE],
    },
    "group": {
        "intuition": "组合因子：把若干成员因子合成为一个排序，检验的是「组合是否比单因子更好」。",
        "reading": "综合分越高代表该风格象限越突出。",
        "example": "价值组里 PE 12 倍、PB 1.0 的股票综合分靠前。",
        "caveats": "组合会掩盖成员之间的分歧；"
                   "组合因子的有效性必须用增量检验（ΔIC）而不是绝对 IC 来判断。",
        "refs": [REF_MULTIPLE_TESTING],
    },
    "alpha101": {
        "intuition": "WorldQuant《101 Formulaic Alphas》的成员，是为多空组合设计的短持有期信号。",
        "reading": "排序型因子，按截面分组使用。",
        "example": "本仓库只登记了编号，公式见官方论文与 Qlib 实现。",
        "caveats": "原始持有期只有 0.6~6.4 天；"
                   "论文只给表达式不给业务注释，各家算子实现可能不同（ts_rank 的横截面/时序、Pearson/Spearman）。",
        "refs": [REF_ALPHA101_PAPER, REF_QLIB_SRC],
    },
}


def primer_for(factor_id: str, operator: str = None, family: str = None,
               window: int = None) -> Dict[str, Any]:
    """Resolve the primer for one factor, substituting the window.

    Lookup order: exact factor id -> operator template -> family fallback.
    Returns a dict with ``scope`` telling which of the three hit.
    """
    if factor_id in FACTOR_PRIMERS:
        return {"scope": "factor", **_render(FACTOR_PRIMERS[factor_id], window)}
    if operator and operator in OPERATOR_PRIMERS:
        return {"scope": "factor", **_render(OPERATOR_PRIMERS[operator], window)}
    if family and family in FAMILY_PRIMERS:
        return {"scope": "family", **_render(FAMILY_PRIMERS[family], window)}
    return {"scope": "none", "intuition": "", "reading": "", "example": "",
            "caveats": "", "refs": []}


def _render(primer: Dict[str, Any], window: int = None) -> Dict[str, Any]:
    resolved = str(window) if window is not None else "N"
    out: Dict[str, Any] = {}
    for key, value in primer.items():
        if key == "refs":
            out[key] = list(value or [])
        elif isinstance(value, str):
            # A template with no window (should not happen) reads as "N 日".
            out[key] = value.replace("{d}", resolved)
        elif key == "detail":
            out[key] = [
                {item_key: item_value.replace("{d}", resolved)
                 if isinstance(item_value, str) else item_value
                 for item_key, item_value in section.items()}
                for section in (value or [])
            ]
        else:
            out[key] = value
    return out
