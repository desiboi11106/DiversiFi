import io
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import streamlit as st

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(page_title="Portfolio Risk Dashboard – Grow-Lio", layout="wide")
st.title("💼 Portfolio Risk Dashboard")
st.caption("Sharpe, beta, diversification metrics, and risk/return visualizations (Python • Excel • Matplotlib)")

# ------------------------------
# Helpers
# ------------------------------
@st.cache_data
def fetch_prices(tickers, start, end, auto_adjust=True):
    df = yf.download(tickers, start=start, end=end, auto_adjust=auto_adjust, progress=False)
    # yfinance returns:
    #  - MultiIndex cols for multiple tickers
    #  - Single Index cols for a single ticker
    # Normalize to a DataFrame of Close only with columns = tickers
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].copy()
    else:
        close = df[["Close"]].copy()
        close.columns = [tickers[0]]
    close = close.dropna(how="all")
    return close

def clean_weights(raw_tickers, raw_weights):
    tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
    weights = [w.strip() for w in raw_weights.split(",") if w.strip()]
    weights = np.array([float(w) for w in weights], dtype=float)
    if len(tickers) != len(weights):
        raise ValueError("Tickers and weights must have the same count.")
    if (weights < 0).any():
        raise ValueError("Weights must be non-negative.")
    if weights.sum() == 0:
        raise ValueError("Weights sum to 0. Provide non-zero weights.")
    weights = weights / weights.sum()
    return tickers, weights

def ann_metrics(returns, rf=0.00):
    """
    returns: pd.Series (portfolio) or pd.DataFrame (assets) of daily returns
    rf: annual risk-free rate as decimal (e.g., 0.02 for 2%)
    """
    mu_d = returns.mean()
    sd_d = returns.std()
    mu = mu_d * 252
    vol = sd_d * np.sqrt(252)
    sharpe = (mu - rf) / vol.replace(0, np.nan) if isinstance(vol, pd.Series) else (mu - rf) / (vol if vol != 0 else np.nan)
    return mu, vol, sharpe

def portfolio_series(prices: pd.DataFrame, weights: np.ndarray):
    """Return portfolio value series normalized to 1.0 start."""
    norm = prices / prices.iloc[0]
    port = (norm * weights).sum(axis=1)
    return port

def covariance_matrix(returns: pd.DataFrame):
    return returns.cov() * 252  # annualized

def beta_vs_benchmark(asset_returns: pd.DataFrame, bench_returns: pd.Series):
    """
    asset_returns: DataFrame of daily returns for assets
    bench_returns: Series of daily returns for benchmark
    """
    betas = {}
    var_b = bench_returns.var()
    if var_b == 0 or np.isnan(var_b):
        return pd.Series({c: np.nan for c in asset_returns.columns})
    for c in asset_returns.columns:
        cov = np.cov(asset_returns[c].dropna(), bench_returns.dropna())[0, 1]
        betas[c] = cov / var_b
    return pd.Series(betas)

def risk_contributions(weights: np.ndarray, cov: pd.DataFrame):
    """
    Marginal contribution to risk & percent contribution.
    """
    w = np.asarray(weights).reshape(-1, 1)
    port_var = float(w.T @ cov.values @ w)
    if port_var <= 0:
        mcr = np.zeros_like(w.flatten())
        pcr = np.zeros_like(w.flatten())
    else:
        mcr = (cov.values @ w).flatten() / np.sqrt(port_var)  # d(Vol)/d(w_i)
        # Percent contribution to variance:
        pcr = (w.flatten() * (cov.values @ w).flatten()) / port_var
    return mcr, pcr, np.sqrt(port_var)

def diversification_stats(weights: np.ndarray, corr: pd.DataFrame):
    # HHI-based diversification (lower HHI -> more diversified)
    hhi = float(np.sum(np.square(weights)))
    div_score = 1 - hhi  # 0 to ~1
    # average positive correlation
    C = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    n = C.shape[0]
    if n > 1:
        upper = C[np.triu_indices(n, 1)]
        avg_corr = float(np.mean(upper))
    else:
        avg_corr = np.nan
    return hhi, div_score, avg_corr

def random_frontier(returns: pd.DataFrame, n_port=3000, rf=0.00):
    """
    Monte Carlo portfolios for a quick frontier.
    """
    mu, Sigma = returns.mean().values * 252, returns.cov().values * 252
    n = len(mu)
    rr, vv, sh, WW = [], [], [], []
    for _ in range(n_port):
        w = np.random.rand(n)
        w = w / w.sum()
        mu_p = float(w @ mu)
        vol_p = float(np.sqrt(w @ Sigma @ w))
        rr.append(mu_p)
        vv.append(vol_p)
        sh.append((mu_p - rf) / (vol_p if vol_p != 0 else np.nan))
        WW.append(w)
    df = pd.DataFrame({"Return": rr, "Volatility": vv, "Sharpe": sh})
    return df, np.array(WW)

def template_excel():
    """
    Provide a simple Excel template: Sheet 'weights' with Ticker, Weight columns.
    """
    df = pd.DataFrame({"Ticker": ["AAPL", "MSFT", "TSLA"], "Weight": [0.4, 0.4, 0.2]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="weights")
    buf.seek(0)
    return buf

# ------------------------------
# Sidebar inputs
# ------------------------------
st.sidebar.header("Portfolio Inputs")

# Choice: type tickers/weights OR upload excel
mode = st.sidebar.radio("Input Method", ["Manual (tickers & weights)", "Upload Excel (weights)"])

default_start = dt.date(2022, 1, 1)
start = st.sidebar.date_input("Start Date", default_start)
end = st.sidebar.date_input("End Date", dt.date.today())

rf_pct = st.sidebar.number_input("Risk-free (annual, %)", min_value=0.0, max_value=10.0, value=0.0, step=0.10)
rf = rf_pct / 100.0

benchmark = st.sidebar.text_input("Benchmark ticker (for beta)", "^GSPC")

if mode == "Manual (tickers & weights)":
    tickers_raw = st.sidebar.text_input("Tickers (comma-separated)", "AAPL, MSFT, TSLA")
    weights_raw = st.sidebar.text_input("Weights (comma-separated)", "0.4, 0.4, 0.2")
    try:
        tickers, weights = clean_weights(tickers_raw, weights_raw)
    except Exception as e:
        st.error(f"Weight error: {e}")
        st.stop()
else:
    uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx) with sheet 'weights' (Ticker, Weight)", type=["xlsx"])
    st.sidebar.download_button("Download template.xlsx", template_excel(), file_name="portfolio_template.xlsx")
    if uploaded is None:
        st.info("Upload an Excel file or switch to Manual mode.")
        st.stop()
    else:
        try:
            wdf = pd.read_excel(uploaded, sheet_name="weights")
            if not {"Ticker", "Weight"}.issubset(set(wdf.columns)):
                st.error("Excel must contain columns: Ticker, Weight (sheet name: weights).")
                st.stop()
            tickers = [str(t).upper().strip() for t in wdf["Ticker"].tolist()]
            weights = np.array(wdf["Weight"].astype(float).tolist())
            if len(tickers) == 0:
                st.error("No tickers found in Excel.")
                st.stop()
            if weights.sum() == 0:
                st.error("Weights sum to 0. Please provide positive weights.")
                st.stop()
            weights = weights / weights.sum()
        except Exception as e:
            st.error(f"Excel parsing error: {e}")
            st.stop()

# ------------------------------
# Data fetch
# ------------------------------
prices = fetch_prices(tickers, start, end)
if prices.empty:
    st.warning("No price data found for your selection.")
    st.stop()

# Align benchmark
try:
    bench_px = fetch_prices([benchmark], start, end).iloc[:, 0]
except Exception:
    bench_px = None

# Daily returns
rets = prices.pct_change().dropna()
if bench_px is not None and not bench_px.empty:
    bench_rets = bench_px.pct_change().dropna()
else:
    bench_rets = None

# Portfolio series & returns
port_series = portfolio_series(prices, weights)
port_rets = port_series.pct_change().dropna()

# ------------------------------
# Top KPIs
# ------------------------------
colA, colB, colC, colD = st.columns(4)
p_mu, p_vol, p_sharpe = ann_metrics(port_rets, rf=rf)
colA.metric("Portfolio Annual Return", f"{p_mu*100:.2f}%")
colB.metric("Portfolio Annual Volatility", f"{p_vol*100:.2f}%")
colC.metric("Portfolio Sharpe", f"{p_sharpe:.2f}")
colD.metric("Holdings / Sum of Weights", f"{len(tickers)} / {weights.sum():.2f}")

# ------------------------------
# Charts: Price & Rolling Vol
# ------------------------------
st.subheader("📈 Portfolio Value & Rolling Volatility")

c1, c2 = st.columns(2)

with c1:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(port_series.index, port_series.values, label="Portfolio (normalized)")
    ax.set_title("Portfolio Value (Start = 1.0)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig, use_container_width=True)

with c2:
    roll = port_rets.rolling(21).std() * np.sqrt(252)
    fig2, ax2 = plt.subplots(figsize=(6.5, 4))
    ax2.plot(roll.index, roll.values)
    ax2.set_title("Rolling 1M Volatility (Annualized)")
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2, use_container_width=True)

# ------------------------------
# Asset metrics (Return / Vol / Sharpe) & Beta
# ------------------------------
st.subheader("📊 Asset Metrics")
asset_mu, asset_vol, asset_sh = ann_metrics(rets, rf=rf)
metrics_df = pd.DataFrame({
    "Ann Return": asset_mu,
    "Ann Volatility": asset_vol,
    "Sharpe": asset_sh
})
if bench_rets is not None and not bench_rets.empty:
    betas = beta_vs_benchmark(rets, bench_rets)
    metrics_df["Beta vs " + benchmark] = betas
st.dataframe(metrics_df.style.format({
    "Ann Return": "{:.2%}", "Ann Volatility": "{:.2%}", "Sharpe": "{:.2f}",
    **({f"Beta vs {benchmark}": "{:.2f}"} if bench_rets is not None else {})
}), use_container_width=True)

# ------------------------------
# Risk contributions & Diversification
# ------------------------------
st.subheader("🧩 Risk Decomposition & Diversification")

cov = covariance_matrix(rets)
corr = rets.corr()
mcr, pcr, port_vol_ann = risk_contributions(weights, cov)
hhi, div_score, avg_corr = diversification_stats(weights, corr)

r1, r2, r3 = st.columns(3)
r1.metric("Portfolio Volatility", f"{port_vol_ann*100:.2f}%")
r2.metric("Diversification Score (1-HHI)", f"{div_score:.3f}")
r3.metric("Avg Pairwise Correlation", f"{avg_corr:.2f}" if not np.isnan(avg_corr) else "—")

# Bar: Percent risk contribution
fig3, ax3 = plt.subplots(figsize=(7, 4))
ax3.bar(tickers, pcr * 100.0)
ax3.set_ylabel("% of Portfolio Variance")
ax3.set_title("Percent Contribution to Risk")
ax3.yaxis.set_major_formatter(PercentFormatter(100))
ax3.grid(True, axis="y", alpha=0.3)
st.pyplot(fig3, use_container_width=True)

# Heatmap: Correlation
fig4, ax4 = plt.subplots(figsize=(6, 5))
im = ax4.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
ax4.set_xticks(range(len(tickers))); ax4.set_xticklabels(tickers, rotation=45, ha="right")
ax4.set_yticks(range(len(tickers))); ax4.set_yticklabels(tickers)
ax4.set_title("Correlation Heatmap")
cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
st.pyplot(fig4, use_container_width=True)

# ------------------------------
# Efficient frontier (Monte Carlo)
# ------------------------------
st.subheader("🌈 Efficient Frontier (Monte Carlo)")
n_sims = st.slider("Number of random portfolios", 1000, 10000, 3000, step=1000)
frontier_df, _ = random_frontier(rets, n_port=n_sims, rf=rf)

fig5, ax5 = plt.subplots(figsize=(7, 5))
ax5.scatter(frontier_df["Volatility"], frontier_df["Return"], s=8, alpha=0.3)
ax5.scatter([p_vol], [p_mu], c="red", s=80, label="Your Portfolio")
ax5.set_xlabel("Volatility"); ax5.set_ylabel("Return"); ax5.set_title("Risk/Return Cloud")
ax5.grid(True, alpha=0.3); ax5.legend()
st.pyplot(fig5, use_container_width=True)

# ------------------------------
# Downloads (metrics + prices)
# ------------------------------
st.subheader("⬇️ Export")
exp1 = metrics_df.copy()
exp1.index.name = "Ticker"
csv_metrics = exp1.to_csv().encode("utf-8")
st.download_button("Download Asset Metrics (CSV)", csv_metrics, "asset_metrics.csv", "text/csv")

csv_prices = prices.to_csv().encode("utf-8")
st.download_button("Download Prices (CSV)", csv_prices, "prices.csv", "text/csv")

# ------------------------------
# Notes / Resume bullets
# ------------------------------
with st.expander("📝 What this dashboard demonstrates (resume-ready)"):
    st.markdown("""
- **Modeled portfolio Sharpe ratio, beta, and diversification** (HHI, correlations, risk contributions).
- **Visualized risk/return trade-offs**: efficient frontier via Monte Carlo, portfolio vs. cloud.
- **Supports Excel inputs** for practical portfolio allocations (sheet: `weights`).
- Built with **Python, Excel I/O, Matplotlib**, and **Streamlit** for interactive UX.
""")
