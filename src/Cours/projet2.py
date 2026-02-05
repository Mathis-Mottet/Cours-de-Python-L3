import yfinance as yf
import pandas as pd
import numpy as np

# =========== paramètres ===========
tickers = ["AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA","META","JPM","V","JNJ",
           "DIS","NFLX","ADBE","PYPL","INTC","CSCO","ORCL","CRM","BABA","KO"]
start_date = "2003-01-01"
end_date = "2026-01-01"
window = 252                 # fenêtre pour rolling Sharpe (1 an ~ 252 jours)
initial_cash = 100_000.0
fee_rate = 0.001             # 0.1% transaction fee per trade (apply to traded value)
max_position_pct = 0.10      # max 10% du portefeuille dans chaque actif
buy_threshold = 0.5          # seuil Sharpe pour acheter
sell_threshold = -0.5        # seuil Sharpe pour vendre (liquidate)
# On trade mensuellement (1er jour de chaque mois present in dates)
# ================================================================

# =========== Téléchargement des prix ===========
raw = yf.download(tickers, start=start_date, end=end_date, group_by="ticker", threads=True)

# Robust extract close_df with one column per ticker
if isinstance(raw.columns, pd.MultiIndex):
    try:
        if 'Close' in raw.columns.get_level_values(1):
            close_df = raw.xs('Close', axis=1, level=1)
        else:
            close_cols = [c for c in raw.columns if (isinstance(c, tuple) and 'Close' in c)]
            close_df = raw.loc[:, close_cols]
            close_df.columns = [c[-1] if isinstance(c, tuple) else c for c in close_df.columns]
    except Exception:
        if ('Adj Close' in raw.columns.get_level_values(1)):
            close_df = raw.xs('Adj Close', axis=1, level=1)
        else:
            raise RuntimeError("Format inattendu des colonnes yfinance")
else:
    if 'Close' in raw.columns:
        close_df = raw['Close']
    elif 'Adj Close' in raw.columns:
        close_df = raw['Adj Close']
    else:
        close_df = raw

# Ensure columns order matches tickers and fill missing
close_df = close_df.reindex(columns=tickers, fill_value=np.nan)
close_df = close_df.ffill().bfill()
close_df = close_df.sort_index()
dates = close_df.index
n_days = len(dates)

# =========== BUY-AND-HOLD 20 TITRES ===========
# Allocate initial_cash equally across the 20 tickers at the first available date.
# Fractions of shares are allowed (so we compute float shares).
# first_prices uses the first row of close_df (ffill/bfill ensures values exist).
first_prices = close_df.iloc[0]
n_assets = len(tickers)
per_asset_cash = initial_cash / n_assets

# Protect against zero or NaN prices (shouldn't happen thanks to ffill/bfill, but be safe)
if (first_prices == 0).any() or first_prices.isna().any():
    # find first non-null row to use as basis
    first_valid_index = close_df.apply(lambda col: col.first_valid_index()).min()
    if first_valid_index is not None:
        first_prices = close_df.loc[first_valid_index]
    else:
        raise RuntimeError("Impossible de déterminer les prix initiaux pour buy-and-hold 20 titres.")

# shares (float allowed)
bh_shares = per_asset_cash / first_prices

# daily value series of the equal-weight 20-assets buy-and-hold
portfolio20_buyhold = (close_df * bh_shares).sum(axis=1)  # Series indexed by dates

# =========== Télécharger S&P500 ===========
sp_raw = yf.download("^GSPC", start=start_date, end=end_date)
if isinstance(sp_raw.columns, pd.MultiIndex):
    if 'Close' in sp_raw.columns.get_level_values(1):
        sp_close = sp_raw.xs('Close', axis=1, level=1)
    elif 'Adj Close' in sp_raw.columns.get_level_values(1):
        sp_close = sp_raw.xs('Adj Close', axis=1, level=1)
    else:
        sp_close = sp_raw.iloc[:, 0]
else:
    if 'Close' in sp_raw.columns:
        sp_close = sp_raw['Close']
    elif 'Adj Close' in sp_raw.columns:
        sp_close = sp_raw['Adj Close']
    else:
        sp_close = sp_raw.iloc[:, 0]

sp_close = sp_close.reindex(dates).ffill().bfill()
# base for buy-and-hold (first valid)
first_valid = sp_close.first_valid_index()
if first_valid is None:
    raise RuntimeError("S&P500 data not available for the period")
sp_base = sp_close.loc[first_valid]
sp_buyhold_value = initial_cash * (sp_close / sp_base)

# =========== utilitaires ===========
def rolling_sharpe_from_prices(price_series):
    """
    price_series: pd.Series indexed by date (may include NaN)
    Returns annualized Sharpe (mean_ann / vol_ann) computed from log returns.
    Note: expects at least 2 returns.
    """
    s = price_series.dropna()
    if s.shape[0] < 2:
        return np.nan
    lr = np.log(s / s.shift(1)).dropna()
    if lr.shape[0] < 2:
        return np.nan
    mean_ann = lr.mean() * 252
    vol_ann = lr.std(ddof=1) * np.sqrt(252)
    if vol_ann == 0:
        return np.nan
    return mean_ann / vol_ann

# detect trade days = first available trading day of each month (or first day)
is_trade_day = np.zeros(n_days, dtype=bool)
for i in range(n_days):
    if i == 0:
        is_trade_day[i] = True
    else:
        if dates[i].month != dates[i-1].month:
            is_trade_day[i] = True

# =========== Initialisation portefeuilles ===========
# Multi-asset active portfolio
positions = {t: 0 for t in tickers}        # shares holdings
cash = initial_cash

hist_positions = pd.DataFrame(index=dates, columns=tickers, dtype=int)
hist_cash = pd.Series(index=dates, dtype=float)
hist_portfolio_value = pd.Series(index=dates, dtype=float)

# S&P active portfolio (single asset)
sp_positions = 0
sp_cash = initial_cash
hist_sp_positions = pd.Series(index=dates, dtype=int)
hist_sp_cash = pd.Series(index=dates, dtype=float)
hist_sp_value = pd.Series(index=dates, dtype=float)

# optional: keep a trade log
trade_log = []

# =========== Boucle journalière avec transactions mensuelles ===========
for i, date in enumerate(dates):
    # On calcule la valeur quotidienne (aucun lookahead ici)
    # valeur multi-portfolio
    portfolio_value = cash + sum(positions[t] * close_df.at[date, t] for t in tickers)
    hist_portfolio_value.at[date] = portfolio_value
    hist_cash.at[date] = cash
    for t in tickers:
        hist_positions.at[date, t] = positions[t]

    # S&P active current value
    sp_val = sp_cash + sp_positions * sp_close.at[date]
    hist_sp_value.at[date] = sp_val
    hist_sp_cash.at[date] = sp_cash
    hist_sp_positions.at[date] = sp_positions

    # si ce n'est pas un jour de trading mensuel, on passe
    if not is_trade_day[i]:
        continue

    # pour chaque ticker, on calcule l'indicateur jusqu'à *hier* (pas d'info du jour)
    for ticker in tickers:
        price_today = close_df.at[date, ticker]
        if i == 0:
            continue
        # fenêtre up to previous day: il exclude current day
        window_start = max(0, i - window)
        price_window = close_df[ticker].iloc[window_start:i].dropna()  # up to i-1
        sharpe_prev = rolling_sharpe_from_prices(price_window)

        # compute allowed max position in value
        current_portfolio_value = cash + sum(positions[t] * close_df.at[date, t] for t in tickers)
        max_position_value = max_position_pct * (current_portfolio_value if current_portfolio_value > 0 else initial_cash)

        # decision rules: buy to reach max_position_value if sharpe_prev >= buy_threshold
        #                 sell (liquidate) if sharpe_prev <= sell_threshold
        if np.isnan(sharpe_prev):
            # no signal
            continue
        elif sharpe_prev >= buy_threshold:
            target_value = max_position_value
            target_shares = int(target_value // price_today)
            if target_shares < 0:
                target_shares = 0
            delta_shares = target_shares - positions[ticker]
            if delta_shares > 0:
                # buy delta_shares, but check cash
                cost = delta_shares * price_today
                fee = cost * fee_rate
                total_cost = cost + fee
                # cap by available cash
                affordable = int((cash) // (price_today * (1 + fee_rate)))
                buy_qty = min(delta_shares, affordable)
                if buy_qty > 0:
                    positions[ticker] += buy_qty
                    spent = buy_qty * price_today
                    cash -= spent
                    cash -= spent * fee_rate
                    trade_log.append((date, ticker, "BUY", buy_qty, price_today, spent, spent * fee_rate))
        elif sharpe_prev <= sell_threshold:
            # liquidate entirely
            sell_qty = positions[ticker]
            if sell_qty > 0:
                proceeds = sell_qty * price_today
                fee = proceeds * fee_rate
                cash += proceeds
                cash -= fee
                trade_log.append((date, ticker, "SELL", sell_qty, price_today, proceeds, fee))
                positions[ticker] = 0
        else:
            # no action (between thresholds)
            pass

    # === S&P active trading (single asset) ===
    # compute sharpe for SP up to yesterday
    if i > 0:
        sp_window_start = max(0, i - window)
        sp_price_window = sp_close.iloc[sp_window_start:i].dropna()
        sp_sharpe_prev = rolling_sharpe_from_prices(sp_price_window)
    else:
        sp_sharpe_prev = np.nan

    sp_price_today = sp_close.at[date]

    # same decision logic for SP active: buy to a target % of portfolio (reuse max_position_pct),
    # or liquidate on negative signal
    sp_current_value = sp_cash + sp_positions * sp_price_today
    sp_max_position_value = max_position_pct * (sp_current_value if sp_current_value > 0 else initial_cash)

    if not np.isnan(sp_sharpe_prev):
        if sp_sharpe_prev >= buy_threshold:
            target_value = sp_max_position_value
            target_shares = int(target_value // sp_price_today)
            delta = target_shares - sp_positions
            if delta > 0:
                # buy delta shares, check cash
                cost = delta * sp_price_today
                fee = cost * fee_rate
                affordable = int((sp_cash) // (sp_price_today * (1 + fee_rate)))
                buy_qty = min(delta, affordable)
                if buy_qty > 0:
                    sp_positions += buy_qty
                    sp_cash -= buy_qty * sp_price_today
                    sp_cash -= buy_qty * sp_price_today * fee_rate
                    trade_log.append((date, "^GSPC", "BUY", buy_qty, sp_price_today, buy_qty * sp_price_today, buy_qty * sp_price_today * fee_rate))
        elif sp_sharpe_prev <= sell_threshold:
            # sell everything
            if sp_positions > 0:
                proceeds = sp_positions * sp_price_today
                fee = proceeds * fee_rate
                sp_cash += proceeds
                sp_cash -= fee
                trade_log.append((date, "^GSPC", "SELL", sp_positions, sp_price_today, proceeds, fee))
                sp_positions = 0

# After loop, ensure final history recorded (already recorded at each day start)

# =========== Résultats and export ===========
# recompute final daily series (already done), but ensure types OK
hist_positions = hist_positions.fillna(0).astype(int)
hist_cash = hist_cash.fillna(method='ffill').fillna(initial_cash)
hist_portfolio_value = hist_portfolio_value.fillna(method='ffill').fillna(initial_cash)

hist_sp_positions = hist_sp_positions.fillna(0).astype(int)
hist_sp_cash = hist_sp_cash.fillna(method='ffill').fillna(initial_cash)
hist_sp_value = hist_sp_value.fillna(method='ffill').fillna(initial_cash)

# prepare summary: multi active portfolio, SP active, SP buy-and-hold, 20-tickers buy-and-hold
summary = pd.DataFrame({
    "Cash_MultiActive": hist_cash,
    "PortfolioValue_MultiActive": hist_portfolio_value,
    "Cash_SPActive": hist_sp_cash,
    "PortfolioValue_SPActive": hist_sp_value,
    "SP500_BuyHold": sp_buyhold_value,
    "Portfolio20_BuyHold": portfolio20_buyhold
}, index=dates)

# add a column diff and ratio vs SP buy-and-hold
summary["Excess_vs_SP_buyhold_abs"] = summary["PortfolioValue_MultiActive"] - summary["SP500_BuyHold"]
summary["Excess_vs_SP_buyhold_pct"] = summary["PortfolioValue_MultiActive"] / summary["SP500_BuyHold"] - 1.0

# positions sheet (multi)
positions_sheet = hist_positions.copy()

# SP active sheet
sp_sheet = pd.DataFrame({
    "SP_Positions": hist_sp_positions,
    "SP_Cash": hist_sp_cash,
    "SP_PortfolioValue": hist_sp_value,
    "SP_Close": sp_close
}, index=dates)

# trade log to DataFrame
trades_df = pd.DataFrame(trade_log, columns=["Date", "Ticker", "Side", "Qty", "Price", "Gross", "Fee"])
trades_df["Date"] = pd.to_datetime(trades_df["Date"])

# Export to Excel (or fallback to CSV if openpyxl missing)
try:
    with pd.ExcelWriter("simulation_corrected_with_bh20.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary")
        positions_sheet.to_excel(writer, sheet_name="positions")
        close_df.to_excel(writer, sheet_name="prices")
        sp_sheet.to_excel(writer, sheet_name="sp_active")
        trades_df.to_excel(writer, sheet_name="trades", index=False)
    print("Fichier créé : simulation_corrected_with_bh20.xlsx")
except ModuleNotFoundError:
    summary.to_csv("summary.csv")
    positions_sheet.to_csv("positions.csv")
    close_df.to_csv("prices.csv")
    sp_sheet.to_csv("sp_active.csv")
    trades_df.to_csv("trades.csv")
    print("openpyxl non installé — fichiers CSV créés (summary.csv, positions.csv, ... )")

# =========== Résumé rapide imprimé ===========
def stats_from_series(val_series):
    # CAGR, max drawdown, final value
    start_val = val_series.iloc[0]
    end_val = val_series.iloc[-1]
    years = (val_series.index[-1] - val_series.index[0]).days / 365.25
    cagr = (end_val / start_val) ** (1 / years) - 1 if start_val > 0 and years > 0 else np.nan
    # simple max drawdown
    running_max = val_series.cummax()
    drawdowns = (val_series - running_max) / running_max
    max_dd = drawdowns.min()
    return {"start": start_val, "end": end_val, "CAGR": cagr, "MaxDrawdown": max_dd}

print("=== Résultats résumés ===")
print("Multi Active:", stats_from_series(summary["PortfolioValue_MultiActive"]))
print("SP Active:", stats_from_series(summary["SP500_BuyHold"]))  # corrected line below if needed
print("SP BuyHold:", stats_from_series(summary["SP500_BuyHold"]))
print("Portfolio20 BuyHold:", stats_from_series(summary["Portfolio20_BuyHold"]))
print("Nombre de trades:", len(trades_df))
