# -*- coding: utf-8 -*-
"""
美股助手 - 云端最终版 (SnapDeploy / Render)
整合 6 项功能：
1. 添加股票有效性验证（拒绝无效代码）
2. 修复 Adj Close 兼容问题
3. 仓位页支持手动输入成本/股数 -> 给出具体买卖股数与价位
4. 分析页展示近 1 年财报（营收/净利润/EPS/同比）
5. 按「行业 x 景气阶段」选择最合适估值模型（多个则等权）
6. 不再展示全部模型平均值，只展示最合适模型价格
"""
import ssl
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

import os
from flask import Flask, render_template, jsonify, request

import yfinance as yf
import pandas as pd

app = Flask(__name__)

DEFAULT_SYMBOLS = ["NVDA", "TSLA", "AAPL", "MSFT", "AMD"]


# ============================================================
# 行业 x 景气阶段 -> 最合适估值模型
# ============================================================
SECTOR_BEST_MODEL = {
    'technology': {
        'up':   ['ps_valuation'],
        'flat': ['pe_valuation'],
        'down': ['pb_valuation', 'mean_reversion'],
    },
    'financial services': {
        'up':   ['pb_valuation'],
        'flat': ['pb_valuation'],
        'down': ['pb_valuation', 'ddm'],
    },
    'healthcare': {
        'up':   ['ps_valuation', 'ev_ebitda'],
        'flat': ['pe_valuation'],
        'down': ['ddm'],
    },
    'consumer cyclical': {
        'up':   ['ev_ebitda'],
        'flat': ['pe_valuation', 'ev_ebitda'],
        'down': ['pb_valuation', 'mean_reversion'],
    },
    'consumer defensive': {
        'up':   ['pe_valuation'],
        'flat': ['pe_valuation', 'ddm'],
        'down': ['ddm'],
    },
    'energy': {
        'up':   ['ev_ebitda', 'ps_valuation'],
        'flat': ['ev_ebitda', 'ddm'],
        'down': ['pb_valuation', 'mean_reversion'],
    },
    'basic materials': {
        'up':   ['ev_ebitda'],
        'flat': ['ev_ebitda', 'pb_valuation'],
        'down': ['pb_valuation', 'mean_reversion'],
    },
    'industrials': {
        'up':   ['ev_ebitda'],
        'flat': ['pe_valuation', 'ev_ebitda'],
        'down': ['pb_valuation', 'mean_reversion'],
    },
    'real estate': {
        'up':   ['pb_valuation', 'ev_ebitda'],
        'flat': ['ddm'],
        'down': ['pb_valuation', 'ddm'],
    },
    'utilities': {
        'up':   ['pe_valuation'],
        'flat': ['ddm'],
        'down': ['ddm'],
    },
    'communication services': {
        'up':   ['ps_valuation', 'ev_ebitda'],
        'flat': ['pe_valuation'],
        'down': ['pe_valuation', 'mean_reversion'],
    },
}

MODEL_LABELS = {
    'analyst_target': '分析师目标价',
    'pe_valuation':   'P/E 估值 (EPS×22)',
    'pb_valuation':   'P/B 估值 (净资产×3)',
    'ps_valuation':   'P/S 估值 (每股营收×5)',
    'ev_ebitda':      'EV/EBITDA 估值 (×12)',
    'ddm':            '股息折现 DDM',
    'mean_reversion': '历史均值回归 (250日)',
}

PHASE_LABELS = {
    'up':   '🔺 上升期',
    'flat': '➖ 维持期',
    'down': '🔻 下降期',
}


# ============================================================
# 工具函数
# ============================================================
def _pick_close(df):
    """兼容 yfinance 新版：Adj Close / Close / MultiIndex 列"""
    if df is None or getattr(df, 'empty', True):
        return None
    try:
        cols = df.columns
        if isinstance(cols, pd.MultiIndex):
            df = df.copy()
            df.columns = [c[0] if isinstance(c, tuple) else c for c in cols]
            cols = df.columns
        for name in ('Adj Close', 'Close'):
            if name in cols:
                return df[name].dropna()
    except Exception:
        pass
    return None


def _pick_col(df, name):
    """取指定列（兼容 MultiIndex）"""
    if df is None or getattr(df, 'empty', True):
        return None
    try:
        cols = df.columns
        if isinstance(cols, pd.MultiIndex):
            df = df.copy()
            df.columns = [c[0] if isinstance(c, tuple) else c for c in cols]
        if name in df.columns:
            return df[name].dropna()
    except Exception:
        pass
    return None


def calc_atr(df, period=14):
    """14 日 ATR（真实波幅均值），衡量个股日常波动大小"""
    try:
        h, l, c = _pick_col(df, 'High'), _pick_col(df, 'Low'), _pick_col(df, 'Close')
        if h is None or l is None or c is None or len(c) < period + 1:
            return None
        prev_c = c.shift(1)
        tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return float(atr) if pd.notna(atr) and atr > 0 else None
    except Exception:
        return None


def get_financials(t):
    """最近 1 年（最多 5 个季度）财报：营收、净利润、EPS、营收同比"""
    result = []
    try:
        inc = t.quarterly_income_stmt
        if inc is None or getattr(inc, 'empty', True):
            return result
        all_cols = list(inc.columns)
        cols = all_cols[:5]

        def safe(row_name, col):
            try:
                if row_name in inc.index:
                    v = inc.loc[row_name, col]
                    if pd.notna(v):
                        return float(v)
            except Exception:
                pass
            return None

        for idx, col in enumerate(cols):
            try:
                date_str = col.strftime('%Y-%m-%d') if hasattr(col, 'strftime') else str(col)[:10]
            except Exception:
                date_str = str(col)[:10]

            revenue = safe('Total Revenue', col)
            net_income = safe('Net Income', col)
            eps = safe('Basic EPS', col)
            if eps is None:
                eps = safe('Diluted EPS', col)

            yoy = None
            try:
                if idx + 4 < len(all_cols) and revenue is not None:
                    prev_rev = safe('Total Revenue', all_cols[idx + 4])
                    if prev_rev not in (None, 0):
                        yoy = (revenue - prev_rev) / abs(prev_rev) * 100
            except Exception:
                pass

            result.append({
                'date': date_str,
                'revenue': round(revenue / 1e9, 2) if revenue else None,
                'netIncome': round(net_income / 1e9, 2) if net_income else None,
                'eps': round(eps, 2) if eps is not None else None,
                'yoy': round(yoy, 1) if yoy is not None else None,
            })
        return result
    except Exception:
        return result


def detect_phase(financials):
    """根据最近两期营收同比判断景气阶段"""
    try:
        ys = [f['yoy'] for f in financials if f.get('yoy') is not None]
        if len(ys) < 2:
            return 'flat'
        latest, prev = ys[0], ys[1]
        if latest > prev + 2 and latest > 0:
            return 'up'
        if latest < prev - 2 or latest < 0:
            return 'down'
        return 'flat'
    except Exception:
        return 'flat'


def calc_all_models(t, current_price):
    """计算全部 7 个估值模型，返回 (models_dict, info)"""
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

    models = {}

    v = info.get('targetMeanPrice')
    if v and v > 0:
        models['analyst_target'] = float(v)

    eps = info.get('trailingEps')
    if eps and eps > 0:
        models['pe_valuation'] = float(eps) * 22

    bv = info.get('bookValue')
    if bv and bv > 0:
        models['pb_valuation'] = float(bv) * 3

    rps = info.get('revenuePerShare')
    if rps and rps > 0:
        models['ps_valuation'] = float(rps) * 5

    try:
        ebitda = info.get('ebitda')
        mc = info.get('marketCap')
        if ebitda and ebitda > 0 and mc and mc > 0:
            debt = info.get('totalDebt') or 0
            cash = info.get('totalCash') or 0
            shares = info.get('sharesOutstanding')
            if not shares and current_price:
                shares = mc / current_price
            if shares and shares > 0:
                fair = (float(ebitda) * 12 - float(debt) + float(cash)) / float(shares)
                if fair > 0:
                    models['ev_ebitda'] = float(fair)
    except Exception:
        pass

    try:
        div = info.get('dividendRate')
        if div and div > 0:
            g, r = 0.05, 0.10
            models['ddm'] = float(div) * (1 + g) / (r - g)
    except Exception:
        pass

    metrics = {'atr': None, 'atrPct': None, 'low52': None}
    try:
        h = yf.download(t.ticker, period="1y", progress=False)
        s = _pick_close(h)
        if s is not None and len(s) > 0:
            avg = float(s.mean())
            if avg > 0:
                models['mean_reversion'] = avg
            try:
                metrics['low52'] = float(s.min())
            except Exception:
                pass
        a = calc_atr(h)
        if a and a > 0:
            metrics['atr'] = a
            metrics['atrPct'] = round(a / current_price * 100, 2) if current_price else None
    except Exception:
        pass

    # ---------- 异常值防护：单模型价格须落在现价的合理区间 ----------
    # BRK-B 曾出现某模型返回"总市值量级"的数字（百万美元），导致 FV 被污染，
    # 进而使止损价 = FV×0.85 变成 133 万。这里直接剔除离谱模型。
    outliers = {}
    if current_price and current_price > 0:
        lo, hi = current_price * 0.2, current_price * 5
        for k in list(models.keys()):
            v = models.get(k)
            if v is None or v <= 0 or v < lo or v > hi:
                outliers[k] = round(float(v), 2) if v else None
                models.pop(k, None)

    return models, info, metrics, outliers


def pick_best_model(sector, phase, models, current_price=None):
    """按行业+阶段挑最合适模型；多个则等权。返回 (fair_value, keys_used, fallback)"""
    sector_key = (sector or '').strip().lower()
    mapping = SECTOR_BEST_MODEL.get(sector_key, {})
    want_keys = mapping.get(phase) or mapping.get('flat') or []

    def _clamp(v):
        if current_price and current_price > 0 and v:
            return max(current_price * 0.3, min(float(v), current_price * 3))
        return v

    usable = [k for k in want_keys if k in models and models.get(k)]
    if usable:
        val = sum(models[k] for k in usable) / len(usable)
        return _clamp(val), usable, False

    # 回退：分析师目标价
    if models.get('analyst_target'):
        return _clamp(models['analyst_target']), ['analyst_target'], True
    # 再回退：任一可用模型
    if models:
        k = list(models.keys())[0]
        return _clamp(models[k]), [k], True
    return None, [], True


# ============================================================
# 页面
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ============================================================
# API: 行情列表
# ============================================================
@app.route("/api/stocks", methods=["POST"])
def get_stocks():
    symbols = (request.json or {}).get("symbols") or DEFAULT_SYMBOLS
    results = []
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="5d")
            if hist is None or hist.empty:
                results.append({"symbol": sym, "error": "无数据"})
                continue

            latest_price = float(hist["Close"].iloc[-1])
            prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else latest_price
            chg = (latest_price - prev_close) / prev_close * 100 if prev_close else 0

            revenue = net_income = eps_surprise = None
            try:
                inc = t.quarterly_income_stmt
                if inc is not None and not inc.empty:
                    if "Total Revenue" in inc.index:
                        revenue = float(inc.loc["Total Revenue"].iloc[0]) / 1e9
                    if "Net Income" in inc.index:
                        net_income = float(inc.loc["Net Income"].iloc[0]) / 1e9
            except Exception:
                pass
            try:
                eh = t.earnings_history
                if eh is not None and not eh.empty:
                    last = eh.iloc[0]
                    if "surprisePercent" in last and pd.notna(last["surprisePercent"]):
                        eps_surprise = float(last["surprisePercent"])
            except Exception:
                pass

            results.append({
                "symbol": sym,
                "price": round(latest_price, 2),
                "change": round(chg, 2),
                "revenue": round(revenue, 2) if revenue else None,
                "netIncome": round(net_income, 2) if net_income else None,
                "epsSurprise": round(eps_surprise, 1) if eps_surprise is not None else None,
            })
        except Exception as e:
            results.append({"symbol": sym, "error": str(e)})

    return jsonify(results)


# ============================================================
# API: 验证股票代码是否有效
# ============================================================
@app.route("/api/validate/<symbol>")
def validate_symbol(symbol):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="5d")
        if hist is None or hist.empty:
            return jsonify({"valid": False, "reason": "无行情数据"})
        return jsonify({"valid": True, "price": round(float(hist["Close"].iloc[-1]), 2)})
    except Exception as e:
        return jsonify({"valid": False, "reason": str(e)})


# ============================================================
# API: 分析（估值 + 财报 + 百分位）
# ============================================================
@app.route("/api/analysis/<symbol>")
def analyze(symbol):
    try:
        t = yf.Ticker(symbol)
        hist5 = t.history(period="5d")
        if hist5 is None or hist5.empty:
            return jsonify({"error": "无行情数据"})
        current_price = float(hist5["Close"].iloc[-1])

        models, info, metrics, outliers = calc_all_models(t, current_price)
        financials = get_financials(t)
        phase = detect_phase(financials)
        sector = info.get('sector') or ''

        fair_value, best_keys, is_fallback = pick_best_model(sector, phase, models, current_price)

        # 历史百分位（2年）
        hist = yf.download(symbol, period="2y", progress=False)
        series = _pick_close(hist)
        if series is None or len(series) == 0:
            return jsonify({"error": "无历史数据"})

        current = float(series.iloc[-1])
        hist_min = float(series.min())
        hist_max = float(series.max())
        position_pct = (current - hist_min) / (hist_max - hist_min) * 100 if hist_max > hist_min else 50
        percentile = float(series.rank(pct=True).iloc[-1]) * 100

        model_display = {k: round(v, 2) for k, v in models.items()}
        discount = round((fair_value - current) / fair_value * 100, 1) if fair_value else None

        best_names = [MODEL_LABELS.get(k, k) for k in best_keys]

        return jsonify({
            "symbol": symbol,
            "sector": sector or "未知",
            "phase": phase,
            "phaseLabel": PHASE_LABELS.get(phase, '➖ 维持期'),
            "currentPrice": round(current, 2),
            "fairValue": round(fair_value, 2) if fair_value else None,
            "bestKeys": best_keys,
            "bestNames": best_names,
            "bestLabel": " + ".join(best_names) if best_names else "—",
            "isFallback": is_fallback,
            "discount": discount,
            "histMin": round(hist_min, 2),
            "histMax": round(hist_max, 2),
            "positionPct": round(position_pct, 1),
            "percentile": round(percentile, 1),
            "models": model_display,
            "financials": financials,
            "outliers": outliers,
            "atrPct": metrics.get('atrPct'),
        })
    except Exception as e:
        return jsonify({"error": str(e)})


# ============================================================
# 板块 ETF 映射（用于板块轮动动量判断）
# ============================================================
SECTOR_ETF = {
    'technology': 'XLK',
    'financial services': 'XLF',
    'healthcare': 'XLV',
    'consumer cyclical': 'XLY',
    'consumer defensive': 'XLP',
    'energy': 'XLE',
    'basic materials': 'XLB',
    'industrials': 'XLI',
    'real estate': 'XLRE',
    'utilities': 'XLU',
    'communication services': 'XLC',
}


def get_market_context(needed_sectors=None):
    """获取可量化的宏观代理指标：大盘趋势、VIX、板块动量排名"""
    ctx = {
        'sp500_price': None, 'sp500_ma200': None, 'sp500_above_ma200': None,
        'vix': None, 'sector_rank': {}, 'sector_weak': False,
    }

    # 1) 标普500 vs 200日均线
    try:
        h = yf.download('^GSPC', period='1y', progress=False)
        s = _pick_close(h)
        if s is not None and len(s) > 200:
            price = float(s.iloc[-1])
            ma200 = float(s.rolling(200).mean().iloc[-1])
            ctx['sp500_price'] = round(price, 2)
            ctx['sp500_ma200'] = round(ma200, 2)
            ctx['sp500_above_ma200'] = price > ma200
    except Exception:
        pass

    # 2) 恐慌指数 VIX
    try:
        v = yf.download('^VIX', period='1mo', progress=False)
        sv = _pick_close(v)
        if sv is not None and len(sv) > 0:
            ctx['vix'] = round(float(sv.iloc[-1]), 2)
    except Exception:
        pass

    # 3) 板块动量（近3个月收益率排名）
    try:
        keys = needed_sectors or list(SECTOR_ETF.keys())
        rets = {}
        for key in keys:
            etf = SECTOR_ETF.get(key)
            if not etf:
                continue
            try:
                hh = yf.Ticker(etf).history(period='6mo')
                ss = _pick_close(hh)
                if ss is not None and len(ss) > 63:
                    r = (float(ss.iloc[-1]) / float(ss.iloc[-63]) - 1) * 100
                    rets[key] = r
            except Exception:
                pass
        if rets:
            ordered = sorted(rets.items(), key=lambda x: x[1], reverse=True)
            total = len(ordered)
            for i, (k, r) in enumerate(ordered):
                ctx['sector_rank'][k] = {'rank': i + 1, 'total': total, 'return3m': round(r, 1)}
    except Exception:
        pass

    return ctx


def calc_cash_ratio(ctx, holding_value, total_capital, override=None):
    """计算建议现金比例(%)：基准20% + 可量化代理指标调整，限制10%~70%"""
    if override is not None:
        try:
            v = float(override)
            v = max(0.0, min(v, 100.0))
            return round(v, 1), [{'item': '手动覆盖', 'delta': None,
                                  'reason': f'你手动设定现金比例为 {v}%'}]
        except Exception:
            pass

    ratio = 20.0
    reasons = []

    # 大盘趋势
    above = ctx.get('sp500_above_ma200')
    sp, ma = ctx.get('sp500_price'), ctx.get('sp500_ma200')
    if above is False:
        ratio += 15
        reasons.append({'item': '大盘趋势', 'delta': 15,
                        'reason': f'标普500({sp}) 低于200日均线({ma})，熊市信号，增加防守'})
    elif above is True:
        reasons.append({'item': '大盘趋势', 'delta': 0,
                        'reason': f'标普500({sp}) 在200日均线上方，趋势健康'})
    else:
        reasons.append({'item': '大盘趋势', 'delta': 0, 'reason': '大盘数据不可用，未调整'})

    # 恐慌指数 VIX
    vix = ctx.get('vix')
    if vix is not None:
        if vix > 30:
            ratio -= 10
            reasons.append({'item': '恐慌指数 VIX', 'delta': -10,
                            'reason': f'VIX={vix}(>30，市场恐慌)，别人恐惧我贪婪，减少现金'})
        elif vix < 15:
            ratio += 10
            reasons.append({'item': '恐慌指数 VIX', 'delta': 10,
                            'reason': f'VIX={vix}(<15，过度乐观)，风险积聚，增加现金'})
        else:
            reasons.append({'item': '恐慌指数 VIX', 'delta': 0,
                            'reason': f'VIX={vix}(15~30 正常区间)，未调整'})
    else:
        reasons.append({'item': '恐慌指数 VIX', 'delta': 0, 'reason': 'VIX 数据不可用，未调整'})

    # 板块轮动（持仓/关注板块动量是否偏弱）
    if ctx.get('sector_weak'):
        ratio += 5
        reasons.append({'item': '板块轮动', 'delta': 5,
                        'reason': '你所持板块动量排名靠后（后1/3），板块退潮，增加现金'})
    else:
        reasons.append({'item': '板块轮动', 'delta': 0, 'reason': '板块动量未处于弱势区间'})

    # 持仓水位
    if total_capital > 0:
        pos_ratio = holding_value / total_capital * 100
        if pos_ratio > 70:
            ratio += 10
            reasons.append({'item': '持仓水位', 'delta': 10,
                            'reason': f'当前持仓占总资金 {round(pos_ratio,1)}%(>70%)，仓位过重，增加现金'})
        elif pos_ratio < 30:
            ratio -= 5
            reasons.append({'item': '持仓水位', 'delta': -5,
                            'reason': f'当前持仓占总资金 {round(pos_ratio,1)}%(<30%)，轻仓可加码，减少现金'})
        else:
            reasons.append({'item': '持仓水位', 'delta': 0,
                            'reason': f'当前持仓占总资金 {round(pos_ratio,1)}%(30%~70% 合理)，未调整'})

    ratio = max(10.0, min(70.0, ratio))
    return round(ratio, 1), reasons


def build_buy_ladder(price, fair_value, amount, atr=None):
    """
    3 档阶梯买入 = 估值闸门(FV安全边际) ∧ 波动闸门(ATR间距)，取更低者。
    档1：便宜就现在买（触发=min(FV×0.9, 现价)），否则挂 FV×0.9 等回调
    档2/3：min(FV×0.8/×0.65, 现价−3ATR/−5ATR) —— 档距由个股波动率自动撑开
    资金配比：正金字塔 20/30/50（越跌买越多，摊低成本）
    """
    if amount <= 0 or price <= 0:
        return []
    atr = atr if (atr and atr > 0) else price * 0.02   # 缺省按 2% 波动
    fv = fair_value if (fair_value and fair_value > 0) else price

    margins = [0.90, 0.80, 0.65]      # 估值锚：安全边际 10%/20%/35%
    atr_mults = [0.0, 3.0, 5.0]       # 波动锚：档1 不额外下探，档2/3 拉开 3/5 倍 ATR
    ratios = [0.20, 0.30, 0.50]       # 越跌买越多
    deep_value = price <= fv * 0.65   # 已处深度价值区 → 档1 立即建仓

    ladder = []
    for i, (m, am, r) in enumerate(zip(margins, atr_mults, ratios)):
        fv_price = round(fv * m, 2)
        atr_price = round(price - am * atr, 2)
        if i == 0:
            trigger = price if deep_value else min(fv_price, price)
        else:
            trigger = min(fv_price, atr_price)
        trigger = round(max(trigger, price * 0.3), 2)   # 兜底不低于现价 30%
        amt = amount * r
        sh = int(amt // trigger) if trigger > 0 else 0
        ladder.append({
            'level': i + 1,
            'trigger': trigger,
            'ratio': round(r * 100),
            'amount': round(amt, 2),
            'shares': sh,
            'fvAnchor': fv_price,
            'atrAnchor': atr_price,
            'gapPct': round((trigger / price - 1) * 100, 1),
            'immediate': price <= trigger + 1e-9,   # 现价是否已触发
        })
    return ladder


def build_sell_ladder(fair_value, shares, price=None, atr=None):
    """
    3 档阶梯卖出 = 估值锚(FV溢价) ∧ 波动闸门(ATR)，取更高者。
    档1：现价已超 FV 则立即减 1/3，否则挂 FV×1.00
    档2/3：max(FV×1.15/×1.30, 现价+3ATR/+5ATR) —— 避免日内噪音洗出
    """
    if not fair_value or shares <= 0:
        return []
    price = price or fair_value
    atr = atr if (atr and atr > 0) else price * 0.02
    fv = fair_value

    mults = [1.00, 1.15, 1.30]
    atr_mults = [0.0, 3.0, 5.0]
    ladder = []
    remaining = shares
    for i, (m, am) in enumerate(zip(mults, atr_mults)):
        fv_price = round(fv * m, 2)
        atr_price = round(price + am * atr, 2)
        trigger = max(fv_price, atr_price) if i > 0 else max(fv_price, min(price, fv_price))
        if i == 0:
            trigger = fv_price if price < fv_price else price   # 已溢价则立即减
        n = shares // 3 if i < len(mults) - 1 else remaining
        n = min(n, remaining)
        if n <= 0:
            continue
        ladder.append({
            'level': i + 1,
            'trigger': round(trigger, 2),
            'ratio': round(n / shares * 100),
            'shares': n,
            'fvAnchor': fv_price,
            'atrAnchor': atr_price,
            'gapPct': round((trigger / price - 1) * 100, 1) if price else 0,
            'immediate': price >= trigger - 1e-9,
        })
        remaining -= n
    return ladder


# ============================================================
# API: 仓位与调整建议（现金比例 + 阶梯买卖）
# ============================================================
@app.route("/api/position", methods=["POST"])
def position():
    data = request.json or {}
    total_capital = float(data.get("capital") or 0)
    symbols = data.get("symbols") or DEFAULT_SYMBOLS
    holdings = data.get("holdings") or {}
    cash_override = data.get("cashOverride")

    # ---------- 1. 抓取每只股票的基础数据 ----------
    items = []
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="5d")
            if hist is None or hist.empty:
                continue
            price = float(hist["Close"].iloc[-1])
            models, info, metrics, outliers = calc_all_models(t, price)
            financials = get_financials(t)
            phase = detect_phase(financials)
            sector = info.get('sector') or ''
            fair_value, best_keys, _ = pick_best_model(sector, phase, models, price)
            if not fair_value:
                fair_value = price * 1.1
            discount = (fair_value - price) / fair_value if fair_value else 0

            h = holdings.get(sym) or {}
            try:
                sh = int(float(h.get('shares'))) if h.get('shares') not in (None, '') else 0
            except Exception:
                sh = 0
            try:
                cost = float(h.get('cost')) if h.get('cost') not in (None, '') else None
            except Exception:
                cost = None

            items.append({
                'symbol': sym, 'price': price, 'fairValue': fair_value,
                'discount': discount, 'sector': sector, 'phase': phase,
                'bestKeys': best_keys, 'shares': sh, 'cost': cost,
                'atr': metrics.get('atr'), 'atrPct': metrics.get('atrPct'),
                'low52': metrics.get('low52'), 'outliers': outliers,
            })
        except Exception:
            continue

    # ---------- 2. 当前持仓市值 ----------
    holding_value = sum(it['price'] * it['shares'] for it in items if it['shares'] > 0)

    # ---------- 3. 市场上下文 + 建议现金比例 ----------
    needed = list({(it['sector'] or '').lower() for it in items if it['sector']})
    ctx = get_market_context(needed)
    pcts = []
    for it in items:
        r = ctx.get('sector_rank', {}).get((it['sector'] or '').lower())
        if r:
            pcts.append(r['rank'] / r['total'])
    ctx['sector_weak'] = bool(pcts) and (sum(pcts) / len(pcts) > 2 / 3)

    cash_ratio, cash_reasons = calc_cash_ratio(ctx, holding_value, total_capital, cash_override)

    cash_amount = total_capital * cash_ratio / 100
    target_position_value = total_capital - cash_amount          # 目标总持仓市值
    new_money = max(0.0, target_position_value - holding_value)  # 可新增投入

    # ---------- 4. 先算各股目标金额，若超预算则等比缩放 ----------
    desired = {}
    for it in items:
        price, fv, discount = it['price'], it['fairValue'], it['discount']
        weight = 0.05 * (1 + discount * 2)
        weight = max(0.01, min(weight, 0.15))
        it['weight'] = weight
        it['targetAmount'] = total_capital * weight
        # 止损 = 论点破坏价：成本−15% / FV×0.85 / 52周低点 / 现价−2ATR，取最低（最宽松）
        atr = it.get('atr') or price * 0.02
        cands = [x for x in [
            it['cost'] * 0.85 if it['cost'] else None,
            fv * 0.85,
            it.get('low52'),
            price - 2 * atr,
        ] if x and x > 0]
        it['stopLoss'] = min(cands) if cands else fv * 0.85

        holding_mv = price * it['shares'] if it['shares'] > 0 else 0.0
        if it['shares'] > 0:
            overvalued = price > fv
            if overvalued or (it['cost'] and (price - it['cost']) / it['cost'] > 0.25):
                desired[it['symbol']] = 0.0      # 卖出场景，不占用买入预算
            elif discount > 0.10 and holding_mv < it['targetAmount']:
                desired[it['symbol']] = it['targetAmount'] - holding_mv
            else:
                desired[it['symbol']] = 0.0
        else:
            desired[it['symbol']] = it['targetAmount'] if discount > 0.05 else 0.0

    total_desired = sum(desired.values())
    scale = 1.0
    if total_desired > new_money > 0:
        scale = new_money / total_desired
    elif new_money <= 0:
        scale = 0.0

    # ---------- 5. 生成逐股计划 ----------
    results = []
    plan_buy_total = 0.0
    for it in items:
        price, fv, discount = it['price'], it['fairValue'], it['discount']
        sh, cost = it['shares'], it['cost']
        holding_mv = price * sh if sh > 0 else 0.0

        item = {
            'symbol': it['symbol'],
            'price': round(price, 2),
            'fairValue': round(fv, 2),
            'discount': round(discount * 100, 1),
            'sector': it['sector'] or '未知',
            'phase': it['phase'],
            'phaseLabel': PHASE_LABELS.get(it['phase'], '➖ 维持期'),
            'weight': round(it['weight'] * 100, 1),
            'targetAmount': round(it['targetAmount'], 2),
            'stopLoss': round(it['stopLoss'], 2),
            'atrPct': it.get('atrPct'), 'low52': it.get('low52'),
            'shares': sh,
            'cost': cost,
            'holdingValue': round(holding_mv, 2),
        }
        sr = ctx.get('sector_rank', {}).get((it['sector'] or '').lower())
        if sr:
            item['sectorRank'] = sr

        if sh > 0:
            pnl = (price - cost) * sh if cost else 0
            pnl_pct = (price - cost) / cost * 100 if cost else 0
            item['pnl'] = round(pnl, 2)
            item['pnlPct'] = round(pnl_pct, 1)

            overvalued = price > fv
            big_profit = cost and (price - cost) / cost > 0.25
            if overvalued or big_profit:
                ladder = build_sell_ladder(fv, sh, price, it.get('atr'))
                item['actionType'] = 'sell'
                item['sellLadder'] = ladder
                item['action'] = '分批止盈（阶梯卖出）'
                item['detail'] = f'现价已接近/超过合理价 {round(fv,2)}，分 {len(ladder)} 档卖出共 {sum(l["shares"] for l in ladder)} 股'
            elif discount > 0.10 and holding_mv < it['targetAmount'] and scale > 0:
                amt = (it['targetAmount'] - holding_mv) * scale
                ladder = build_buy_ladder(price, fv, amt, it.get('atr'))
                plan_buy_total += sum(l['amount'] for l in ladder)
                item['actionType'] = 'buy'
                item['buyLadder'] = ladder
                item['action'] = '低估且低于目标仓位：加仓'
                item['detail'] = f'目标持仓 {round(it["targetAmount"])} 美元，当前 {round(holding_mv)} 美元'
            else:
                item['actionType'] = 'hold'
                item['action'] = '持有观望'
                item['detail'] = f'目标价 {round(fv,2)}，跌破 {round(it["stopLoss"],2)} 止损'
        else:
            if discount > 0.05 and scale > 0:
                amt = it['targetAmount'] * scale
                ladder = build_buy_ladder(price, fv, amt, it.get('atr'))
                plan_buy_total += sum(l['amount'] for l in ladder)
                item['actionType'] = 'buy'
                item['buyLadder'] = ladder
                item['action'] = '低估：分批建仓'
                item['detail'] = f'配置 {round(amt)} 美元（占总资金 {round(it["weight"]*100,1)}%）'
            else:
                item['actionType'] = 'hold'
                item['action'] = '估值偏高/合理：暂不建仓'
                item['detail'] = '等待回调至合理价位以下再考虑'

        results.append(item)

    remain = max(0.0, new_money - plan_buy_total)

    return jsonify({
        'totalCapital': total_capital,
        'cashRatio': cash_ratio,
        'cashAmount': round(cash_amount, 2),
        'cashReasons': cash_reasons,
        'holdingValue': round(holding_value, 2),
        'targetPositionValue': round(target_position_value, 2),
        'newMoney': round(new_money, 2),
        'planBuyTotal': round(plan_buy_total, 2),
        'remainInvestable': round(remain, 2),
        'market': {
            'sp500Price': ctx.get('sp500_price'),
            'sp500Ma200': ctx.get('sp500_ma200'),
            'sp500AboveMa200': ctx.get('sp500_above_ma200'),
            'vix': ctx.get('vix'),
        },
        'items': results,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
