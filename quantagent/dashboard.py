"""Small Streamlit research dashboard."""
import streamlit as st

from .workflow import run_research

st.set_page_config(page_title="QuantAgent", page_icon=None, layout="wide")
st.title("QuantAgent")
st.caption("Agentic Quant Research & Model Evaluation — research-only simulation")
with st.sidebar:
    symbol = st.text_input("Symbol", "DEMO")
    source = st.selectbox("Data source", ["synthetic", "yfinance"])
    periods = st.slider("Synthetic observations", 150, 3000, 800, 100) if source == "synthetic" else 800
    seed = st.number_input("Random seed", min_value=0, max_value=100000, value=7)
    costs = st.slider("Transaction cost (bps)", 0.0, 50.0, 5.0, 1.0)
if st.button("Run research workflow", type="primary"):
    try:
        with st.spinner("Running validated research pipeline..."):
            report = run_research(
                symbol=symbol,
                source=source,
                periods=int(periods),
                seed=int(seed),
                transaction_cost_bps=float(costs),
            )
        a, b, c, d = st.columns(4)
        a.metric("Test accuracy", f"{report['model']['metrics']['accuracy']:.1%}")
        b.metric("Total return", f"{report['backtest']['total_return']:.1%}")
        c.metric("Annualized Sharpe", f"{report['backtest']['annualized_sharpe']:.2f}")
        d.metric("Max drawdown", f"{report['backtest']['max_drawdown']:.1%}")
        st.subheader("Integrity checks")
        st.json(report["integrity_checks"])
        st.subheader("Agent trace")
        st.json(report["agent_trace"])
        st.subheader("Complete experiment report")
        st.json(report)
    except (ValueError, RuntimeError, TypeError) as exc:
        st.error(str(exc))
st.warning(
    "Synthetic data is for pipeline validation only. Historical backtests do not establish future profitability."
)


def run():
    import sys

    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", __file__]
    raise SystemExit(stcli.main())
