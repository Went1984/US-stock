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

    try:
        h = yf.download(t.ticker, period="1y", progress=False)
        s = _pick_close(h)
        if s is not None and len(s) > 0:
            avg = float(s.mean())
            if avg > 0:
                models['mean_reversion'] = avg
    except Exception:
        pass

    return models, info


def pick_best_model(sector, phase, models):
    """按行业+阶段挑最合适模型；多个则等权。返回 (fair_value, keys_used, fallback)"""
    sector_key = (sector or '').strip().lower()
    mapping = SECTOR_BEST_MODEL.get(sector_key, {})
    want_keys = mapping.get(phase) or mapping.get('flat') or []

    usable = [k for k in want_keys if k in models and models.get(k)]
    if usable:
        val = sum(models[k] for k in usable) / len(usable)
        return val, usable, False

    # 回退：分析师目标价
    if models.get('analyst_target'):
        return models['analyst_target'], ['analyst_target'], True
    # 再回退：任一可用模型
    if models:
        k = list(models.keys())[0]
        return models[k], [k], True
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

        models, info = calc_all_models(t, current_price)
        financials = get_financials(t)
        phase = detect_phase(financials)
        sector = info.get('sector') or ''

        fair_value, best_keys, is_fallback = pick_best_model(sector, phase, models)

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
        })
    except Exception as e:
        return jsonify({"error": str(e)})


# ============================================================
# API: 仓位（支持持仓成本/股数）
# ============================================================
@app.route("/api/position", methods=["POST"])
def position():
    data = request.json or {}
    total_capital = float(data.get("capital") or 0)
    symbols = data.get("symbols") or DEFAULT_SYMBOLS
    holdings = data.get("holdings") or {}

    results = []
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="5d")
            if hist is None or hist.empty:
                continue
            price = float(hist["Close"].iloc[-1])

            models, info = calc_all_models(t, price)
            financials = get_financials(t)
            phase = detect_phase(financials)
            sector = info.get('sector') or ''
            fair_value, best_keys, _ = pick_best_model(sector, phase, models)
            if not fair_value:
                fair_value = price * 1.1

            discount = (fair_value - price) / fair_value if fair_value else 0
            stop_loss = fair_value * 0.85
            target_price = fair_value

            h = holdings.get(sym) or {}
            try:
                cost = float(h.get('cost')) if h.get('cost') not in (None, '',) else None
            except Exception:
                cost = None
            try:
                shares = int(float(h.get('shares'))) if h.get('shares') not in (None, '',) else 0
            except Exception:
                shares = 0

            item = {
                "symbol": sym,
                "price": round(price, 2),
                "fairValue": round(fair_value, 2),
                "discount": round(discount * 100, 1),
                "sector": sector or "未知",
                "phase": phase,
                "stopLoss": round(stop_loss, 2),
                "targetPrice": round(target_price, 2),
                "cost": cost,
                "shares": shares,
            }

            if shares > 0 and cost:
                # ===== 已持仓 =====
                market_value = price * shares
                pnl = (price - cost) * shares
                pnl_pct = (price - cost) / cost * 100
                item.update({
                    "marketValue": round(market_value, 2),
                    "pnl": round(pnl, 2),
                    "pnlPct": round(pnl_pct, 1),
                    "holding": True,
                })

                if pnl_pct > 50 and price > fair_value:
                    item["actionType"] = "sell"
                    item["delta"] = shares
                    item["action"] = "大幅盈利且已超合理价：清仓锁定利润"
                    item["detail"] = f"建议卖出全部 {shares} 股，落袋为安"
                elif pnl_pct > 25 and price > fair_value:
                    sell_n = max(1, shares // 3)
                    item["actionType"] = "sell"
                    item["delta"] = sell_n
                    item["action"] = "盈利丰厚且估值偏高：减仓 1/3"
                    item["detail"] = f"建议卖出 {sell_n} 股，保留 {shares - sell_n} 股，剩余仓位止损设 {stop_loss}"
                elif pnl_pct < -20 and discount <= 0.05:
                    item["actionType"] = "sell"
                    item["delta"] = shares
                    item["action"] = "亏损扩大且无估值优势：止损离场"
                    item["detail"] = f"建议卖出全部 {shares} 股，止损价 {stop_loss}"
                elif pnl_pct < -15 and discount > 0.10:
                    buy_n = int((total_capital * 0.05) // price) if price > 0 else 0
                    new_avg = ((cost * shares) + (price * buy_n)) / (shares + buy_n) if buy_n > 0 else cost
                    item["actionType"] = "buy"
                    item["delta"] = buy_n
                    item["action"] = "亏损但显著低估：可补仓摊薄"
                    item["detail"] = f"建议补仓 {buy_n} 股，成本由 {cost} 摊薄至约 {round(new_avg, 2)}" if buy_n > 0 else "资金不足，暂观望"
                else:
                    item["actionType"] = "hold"
                    item["delta"] = 0
                    item["action"] = "持有观望"
                    item["detail"] = f"目标价 {target_price}，跌破 {stop_loss} 止损"
            else:
                # ===== 未持仓 =====
                weight = 0.05 * (1 + discount * 2)
                weight = max(0.01, min(weight, 0.15))
                alloc = total_capital * weight
                buy_n = int(alloc // price) if price > 0 else 0

                if discount > 0.15:
                    act = "强烈买入（深度折价）"
                elif discount > 0.05:
                    act = "分批买入"
                elif discount > -0.05:
                    act = "持有观望（估值合理）"
                else:
                    act = "高估，不建议买入"

                item.update({
                    "holding": False,
                    "actionType": "buy" if discount > 0.05 else "hold",
                    "delta": buy_n if discount > 0.05 else 0,
                    "action": act,
                    "weight": round(weight * 100, 1),
                    "allocation": round(alloc, 2),
                    "detail": f"配置 {round(alloc)} 美元 / 约 {buy_n} 股" if buy_n > 0 else "当前不建议建仓",
                })

            results.append(item)
        except Exception as e:
            results.append({"symbol": sym, "error": str(e)})

    return jsonify(results)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
