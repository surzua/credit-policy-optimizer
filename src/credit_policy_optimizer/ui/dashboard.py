"""Credit Policy Cockpit: Interactive Decision Intelligence & Economic Optimization Dashboard."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from credit_policy_optimizer.api.app import DEFAULT_MODEL_PATH, get_model
from credit_policy_optimizer.data.generator import PortfolioSimulator
from credit_policy_optimizer.decision.economics import (
    CreditPolicyOptimizer,
    EconomicParameters,
    compute_breakeven_pd,
    compute_loan_net_gain,
    compute_loan_net_loss,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# Page & Style Configuration
# ==============================================================================

st.set_page_config(
    page_title="Credit Policy Cockpit | Decision Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
        margin-bottom: 1rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }
    .metric-title {
        font-size: 0.825rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
    }
    .metric-delta-positive {
        color: #10b981;
        font-size: 0.875rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }
    .metric-delta-negative {
        color: #ef4444;
        font-size: 0.875rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }

    /* Badges */
    .badge-approved {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-rejected {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Header banner */
    .cockpit-header {
        background: linear-gradient(90deg, #1e1b4b 0%, #0f172a 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.75rem;
    }
    .cockpit-title {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .cockpit-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 0.35rem;
        margin-bottom: 0;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# Model & Portfolio Caching
# ==============================================================================


@st.cache_resource(show_spinner="Cargando pipeline de riesgo calibrado...")
def load_cached_model() -> Any:
    """Load or lazily initialize the calibrated credit model pipeline."""
    return get_model(DEFAULT_MODEL_PATH)


@st.cache_data(show_spinner="Generando y puntuando portafolio sintético de prueba...")
def generate_scored_portfolio(n_samples: int = 4000, seed: int = 42) -> pl.DataFrame:
    """Generate and score a realistic benchmark portfolio."""
    simulator = PortfolioSimulator(seed=seed)
    df_raw = simulator.simulate(n_samples=n_samples)

    pipeline = load_cached_model()
    # Predict calibrated probability of default
    df_pandas = df_raw.to_pandas()
    probas = pipeline.predict_proba(df_pandas)[:, 1]

    return df_raw.with_columns(pl.Series("calibrated_pd", probas))


# Pre-load data and model
try:
    model_pipeline = load_cached_model()
    df_portfolio = generate_scored_portfolio(n_samples=3500, seed=42)
except Exception as exc:
    st.error(f"Error al inicializar el modelo o portafolio: {exc}")
    st.stop()


# ==============================================================================
# Sidebar: Institutional Macro & Economic Parameters
# ==============================================================================

with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/FastAPI.svg",
        width=48,
    )
    st.title("Parámetros Globales")
    st.caption("Configuración institucional de riesgo y tesorería")

    st.markdown("---")
    st.subheader("🏦 Economía del Crédito")

    cost_of_funds = st.slider(
        "Costo de Fondos Institucional (CoF)",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.005,
        format="%.3f",
        help="Tasa pasiva o costo marginal de fondeo del capital prestado.",
    )

    lgd = st.slider(
        "Severidad de Pérdida (LGD)",
        min_value=0.10,
        max_value=0.95,
        value=0.45,
        step=0.05,
        format="%.2f",
        help="Loss Given Default: porcentaje del principal no recuperable tras el default.",
    )

    interest_rate = st.slider(
        "Tasa de Interés Activa Promedio (APR)",
        min_value=0.05,
        max_value=0.50,
        value=0.18,
        step=0.01,
        format="%.2f",
        help="Tasa nominal anual cobrada al acreditado cumplidor.",
    )

    loan_term = st.selectbox(
        "Plazo Promedio del Préstamo (Meses)",
        options=[6, 12, 18, 24, 36],
        index=1,
    )

    st.markdown("---")
    st.subheader("⚙️ Modo de Ejecución")
    engine_mode = st.radio(
        "Backend de Evaluación",
        options=["⚡ En Memoria (In-Process Polars)", "🔌 API REST (FastAPI Local)"],
        index=0,
    )

    # Calculate global breakeven PD for reference
    global_breakeven_pd = compute_breakeven_pd(
        interest_rate=interest_rate,
        cost_of_funds=cost_of_funds,
        lgd=lgd,
        term_months=loan_term,
    )

    st.info(
        f"**Umbral Breakeven Teórico ($p^*$):** `{global_breakeven_pd * 100:.2f}%`\n\n"
        "Cualquier solicitante con $PD > p^*$ genera un Valor Esperado ($EV$) negativo.",
        icon="ℹ️",
    )


# ==============================================================================
# Header Banner
# ==============================================================================

st.markdown(
    """
    <div class="cockpit-header">
        <h1 class="cockpit-title">Credit Policy Cockpit</h1>
        <p class="cockpit-subtitle">
            Plataforma interactiva de Decision Intelligence para originación crediticia y
            maximización del P&L
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs
tab_roi, tab_whatif, tab_underwriting, tab_stress, tab_api = st.tabs(
    [
        "💼 Business ROI & Impacto",
        "🎛️ Policy Studio (What-If)",
        "👤 Live Underwriting (Sandbox)",
        "🌪️ Stress Testing Macro",
        "🔌 Integración & Arquitectura",
    ]
)


# ==============================================================================
# TAB 1: Business ROI & Impacto Económico
# ==============================================================================

with tab_roi:
    st.markdown("### 📈 El Valor de Negocio: Comparativa de Políticas de Crédito")
    st.write(
        "Demostración del beneficio económico neto (**P&L**) que se obtiene al pasar de una "
        "regla ingenua de clasificación tradicional a una **política económica optimizada**."
    )

    portfolio_volume_millions = st.slider(
        "Volumen de Cartera Originada Objetivo (Millones de USD)",
        min_value=5.0,
        max_value=250.0,
        value=50.0,
        step=5.0,
        format="$%.0fM",
    )

    # Initialize Economic Engine with current sidebar params
    econ_params = EconomicParameters(
        default_lgd=lgd,
        default_cost_of_funds=cost_of_funds,
        default_interest_rate=interest_rate,
        default_loan_term_months=loan_term,
    )
    df_opt_input = df_portfolio.with_columns(pl.col("calibrated_pd").alias("pd"))
    optimizer = CreditPolicyOptimizer(
        portfolio=df_opt_input,
        params=econ_params,
    )

    # Run optimization on the cached portfolio
    opt_result = optimizer.optimize_threshold(grid_size=300, metric="expected_pnl")

    # Scaling factor from sample dataset to target portfolio volume
    sample_exposure = opt_result.optimal_policy.total_exposure
    scale_factor = (portfolio_volume_millions * 1_000_000.0) / max(sample_exposure, 1.0)

    # Scaled metrics
    pnl_opt = opt_result.optimal_policy.expected_pnl * scale_factor
    pnl_base = opt_result.baseline_policy_05.expected_pnl * scale_factor
    delta_pnl = pnl_opt - pnl_base

    pct_pnl_boost = (delta_pnl / max(abs(pnl_base), 1.0)) * 100
    opt_def_rate_pct = opt_result.optimal_policy.expected_default_rate * 100
    base_def_rate_pct = opt_result.baseline_policy_05.expected_default_rate * 100
    opt_roe_pct = opt_result.optimal_policy.return_on_exposure * 100
    base_roe_pct = opt_result.baseline_policy_05.return_on_exposure * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">P&L Neto Optimizado</div>
                <div class="metric-value">${pnl_opt:,.0f}</div>
                <div class="metric-delta-positive">ROE: {opt_roe_pct:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">P&L Política Ingenua (p=0.5)</div>
                <div class="metric-value">${pnl_base:,.0f}</div>
                <div class="metric-delta-negative">ROE: {base_roe_pct:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Ganancia Neta Adicional</div>
                <div class="metric-value" style="color: #10b981;">+${delta_pnl:,.0f}</div>
                <div class="metric-delta-positive">+{pct_pnl_boost:.1f}% vs. status quo</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Tasa de Mora Aprobada</div>
                <div class="metric-value">{opt_def_rate_pct:.2f}%</div>
                <div class="metric-delta-positive">vs {base_def_rate_pct:.2f}% (Ingenua)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Comparación de P&L y Pérdidas por Mora")

    fig_comp = go.Figure()
    categories = ["Política Ingenua (p = 0.50)", "Credit Policy Optimizer (p* Óptimo)"]
    pnl_values = [pnl_base, pnl_opt]
    loss_values = [
        opt_result.baseline_policy_05.expected_loss * scale_factor,
        opt_result.optimal_policy.expected_loss * scale_factor,
    ]

    fig_comp.add_trace(
        go.Bar(
            name="P&L Neto (Ganancia)",
            x=categories,
            y=pnl_values,
            marker_color="#10b981",
            text=[f"${v:,.0f}" for v in pnl_values],
            textposition="auto",
        )
    )
    fig_comp.add_trace(
        go.Bar(
            name="Pérdidas Esperadas por Default",
            x=categories,
            y=loss_values,
            marker_color="#ef4444",
            text=[f"${v:,.0f}" for v in loss_values],
            textposition="auto",
        )
    )

    fig_comp.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    loss_reduction = (
        opt_result.baseline_policy_05.expected_loss - opt_result.optimal_policy.expected_loss
    ) * scale_factor

    st.success(
        f"**Conclusión para el Comité:** Para una cartera originada de "
        f"**${portfolio_volume_millions:.0f}M USD**, el algoritmo de optimización genera un "
        f"incremento de **+${delta_pnl:,.0f} USD** en margen neto, recortando las pérdidas "
        f"por default en **${loss_reduction:,.0f} USD**.",
        icon="💎",
    )


# ==============================================================================
# TAB 2: Policy Studio (What-If)
# ==============================================================================

with tab_whatif:
    st.markdown("### 🎛️ Curva de P&L y Trade-off de Aprobación vs. Riesgo")
    st.write(
        "Al alterar las condiciones económicas en la barra lateral, la curva de P&L se "
        "recalcula en tiempo real para revelar el umbral óptimo de corte ($p^*$) y el "
        "punto de equilibrio financiero."
    )

    # Compute threshold curve across range of cutoffs
    threshold_grid = np.linspace(0.01, 0.60, 60)
    policies = [optimizer.evaluate_policy(float(t)) for t in threshold_grid]
    curve_df = pl.DataFrame([p.model_dump() for p in policies])

    curve_p = curve_df["threshold"].to_numpy()
    curve_pnl = curve_df["expected_pnl"].to_numpy()
    curve_app_rate = (curve_df["approval_rate"] * 100).to_numpy()
    curve_def_rate = (curve_df["expected_default_rate"] * 100).to_numpy()

    # Optimal point
    opt_p = opt_result.optimal_threshold
    opt_pnl_val = opt_result.optimal_policy.expected_pnl

    fig_curve = go.Figure()

    # Area trace for PnL
    fig_curve.add_trace(
        go.Scatter(
            x=curve_p * 100,
            y=curve_pnl,
            mode="lines",
            name="P&L Portafolio ($)",
            line=dict(color="#38bdf8", width=3.5),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.1)",
        )
    )

    # Marker for optimal cutoff
    fig_curve.add_trace(
        go.Scatter(
            x=[opt_p * 100],
            y=[opt_pnl_val],
            mode="markers+text",
            name=f"Umbral Óptimo p* ({opt_p * 100:.2f}%)",
            text=[f"  p* = {opt_p * 100:.2f}% (Máximo P&L)"],
            textposition="top right",
            marker=dict(color="#10b981", size=14, symbol="star"),
        )
    )

    # Vertical line for breakeven PD
    fig_curve.add_vline(
        x=global_breakeven_pd * 100,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text=f"Breakeven ({global_breakeven_pd * 100:.2f}%)",
        annotation_position="bottom right",
    )

    # Vertical line for naive cutoff 50%
    if max(curve_p) >= 0.50:
        fig_curve.add_vline(
            x=50.0,
            line_dash="dot",
            line_color="#ef4444",
            annotation_text="Corte Ingenuo (50%)",
            annotation_position="top right",
        )

    fig_curve.update_layout(
        title="Curva de P&L Esperado vs. Umbral de Corte de Probabilidad",
        xaxis_title="Umbral de Probabilidad de Default (%)",
        yaxis_title="P&L Neto Esperado ($)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_curve, use_container_width=True)

    # Second row: Approval Rate vs Default Rate
    st.markdown("#### Curva de Trade-off: Tasa de Aprobación vs. Tasa de Incumplimiento")
    fig_tradeoff = go.Figure()

    fig_tradeoff.add_trace(
        go.Scatter(
            x=curve_app_rate,
            y=curve_def_rate,
            mode="lines+markers",
            name="Curva Eficiente",
            line=dict(color="#a855f7", width=3),
            marker=dict(size=4),
        )
    )
    fig_tradeoff.add_trace(
        go.Scatter(
            x=[opt_result.optimal_policy.approval_rate * 100],
            y=[opt_result.optimal_policy.expected_default_rate * 100],
            mode="markers+text",
            name="Punto Óptimo de Política",
            text=["  Punto Óptimo"],
            textposition="top left",
            marker=dict(color="#10b981", size=14, symbol="circle-open-dot"),
        )
    )

    fig_tradeoff.update_layout(
        xaxis_title="Tasa de Aprobación (%)",
        yaxis_title="Tasa de Default de la Cartera Aprobada (%)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        height=320,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_tradeoff, use_container_width=True)


# ==============================================================================
# TAB 3: Live Underwriting Sandbox
# ==============================================================================

with tab_underwriting:
    st.markdown("### 👤 Cockpit del Suscriptor: Evaluación Unitaria con Explicabilidad")
    st.write(
        "Evalúe en tiempo real una solicitud individual, observando la **PD calibrada**, "
        "el **desglose financiero de ganancia vs. pérdida** y los **factores clave** de la "
        "decisión crediticia."
    )

    # Archetype selector
    archetypes = {
        "⭐ Cliente Prime (Bajo riesgo)": {
            "income": 9500.0,
            "dti": 0.18,
            "delinq": 0,
            "util": 0.12,
            "amount": 15000.0,
            "term": 12,
            "rate": 0.16,
        },
        "🎓 Joven Profesional (Riesgo Moderado)": {
            "income": 3800.0,
            "dti": 0.32,
            "delinq": 0,
            "util": 0.45,
            "amount": 8000.0,
            "term": 12,
            "rate": 0.20,
        },
        "⚠️ Perfil Subprime Sobreendeudado": {
            "income": 2200.0,
            "dti": 0.58,
            "delinq": 2,
            "util": 0.88,
            "amount": 10000.0,
            "term": 18,
            "rate": 0.24,
        },
        "📈 Comerciante con Utilización Alta": {
            "income": 5500.0,
            "dti": 0.41,
            "delinq": 1,
            "util": 0.79,
            "amount": 12000.0,
            "term": 12,
            "rate": 0.22,
        },
    }

    selected_arch = st.selectbox(
        "Cargar Perfil Arquetípico Predefinido:",
        options=list(archetypes.keys()),
        index=1,
    )
    arch_data = archetypes[selected_arch]

    with st.form("underwriting_form"):
        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            u_income = st.number_input(
                "Ingreso Mensual Bruto (USD)",
                min_value=500.0,
                max_value=50000.0,
                value=float(arch_data["income"]),
                step=100.0,
            )
            u_dti = st.slider(
                "Debt-to-Income Ratio (DTI)",
                min_value=0.01,
                max_value=1.0,
                value=float(arch_data["dti"]),
                step=0.01,
            )

        with col_f2:
            u_delinq = st.number_input(
                "Moras Históricas 30+ Días (últimos 24m)",
                min_value=0,
                max_value=10,
                value=int(arch_data["delinq"]),
                step=1,
            )
            u_util = st.slider(
                "Utilización de Líneas Rotativas",
                min_value=0.01,
                max_value=1.0,
                value=float(arch_data["util"]),
                step=0.01,
            )

        with col_f3:
            u_amount = st.number_input(
                "Monto Solicitado (USD)",
                min_value=500.0,
                max_value=100000.0,
                value=float(arch_data["amount"]),
                step=500.0,
            )
            u_term = st.selectbox(
                "Plazo Solicitado (Meses)",
                options=[6, 12, 18, 24, 36],
                index=[6, 12, 18, 24, 36].index(int(arch_data["term"])),
            )
            u_rate = st.slider(
                "Tasa Activa Ofrecida (APR)",
                min_value=0.05,
                max_value=0.60,
                value=float(arch_data["rate"]),
                step=0.01,
            )

        evaluate_btn = st.form_submit_button(
            "⚡ Evaluar Solicitud con Motor Económico", use_container_width=True
        )

    # Perform evaluation
    applicant_df = pl.DataFrame(
        {
            "application_id": ["APP-LIVE-001"],
            "monthly_income": [u_income],
            "debt_to_income": [u_dti],
            "historical_delinquencies": [u_delinq],
            "revolving_utilization": [u_util],
            "loan_amount": [u_amount],
            "loan_term_months": [u_term],
            "interest_rate": [u_rate],
            "cost_of_funds": [cost_of_funds],
        }
    )

    applicant_pd = float(model_pipeline.predict_proba(applicant_df.to_pandas())[0, 1])

    # Financial economics
    loan_gain = compute_loan_net_gain(
        amount=u_amount,
        interest_rate=u_rate,
        cost_of_funds=cost_of_funds,
        term_months=u_term,
    )
    loan_loss = compute_loan_net_loss(
        amount=u_amount,
        lgd=lgd,
        cost_of_funds=cost_of_funds,
        term_months=u_term,
    )
    expected_value = (1.0 - applicant_pd) * loan_gain - applicant_pd * loan_loss

    indiv_breakeven_pd = compute_breakeven_pd(
        interest_rate=u_rate,
        cost_of_funds=cost_of_funds,
        lgd=lgd,
        term_months=u_term,
    )

    is_approved = expected_value > 0.0

    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.markdown("#### Indicador de Riesgo Calibrado")

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=applicant_pd * 100,
                number={"suffix": "%", "valueformat": ".2f"},
                title={"text": "Probabilidad Calibrada de Default (PD)"},
                delta={
                    "reference": indiv_breakeven_pd * 100,
                    "increasing": {"color": "#ef4444"},
                    "decreasing": {"color": "#10b981"},
                },
                gauge={
                    "axis": {"range": [0, 60], "tickwidth": 1},
                    "bar": {"color": "#ffffff"},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 5], "color": "rgba(16, 185, 129, 0.5)"},
                        {"range": [5, 15], "color": "rgba(234, 179, 8, 0.5)"},
                        {"range": [15, 30], "color": "rgba(249, 115, 22, 0.5)"},
                        {"range": [30, 60], "color": "rgba(239, 68, 68, 0.5)"},
                    ],
                    "threshold": {
                        "line": {"color": "#f59e0b", "width": 4},
                        "thickness": 0.8,
                        "value": indiv_breakeven_pd * 100,
                    },
                },
            )
        )
        fig_gauge.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=20, r=20, t=30, b=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        if is_approved:
            st.markdown(
                '<div class="badge-approved" style="font-size: 1.1rem; padding: 6px 18px;">'
                "✅ DICTAMEN: SOLICITUD APROBADA</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="badge-rejected" style="font-size: 1.1rem; padding: 6px 18px;">'
                "❌ DICTAMEN: SOLICITUD RECHAZADA</div>",
                unsafe_allow_html=True,
            )

        st.caption(
            f"Breakeven personal: **{indiv_breakeven_pd * 100:.2f}%** | "
            f"PD estimada: **{applicant_pd * 100:.2f}%**"
        )

    with res_col2:
        st.markdown("#### Desglose de Utilidad Financiera ($EV$)")

        fig_waterfall = go.Figure(
            go.Waterfall(
                name="Valor Esperado",
                orientation="v",
                measure=["relative", "relative", "total"],
                x=[
                    "Ganancia si Cumple",
                    "Pérdida Esperada Default",
                    "Valor Esperado (EV)",
                ],
                textposition="outside",
                text=[
                    f"+${(1.0 - applicant_pd) * loan_gain:,.1f}",
                    f"-${applicant_pd * loan_loss:,.1f}",
                    f"${expected_value:,.1f}",
                ],
                y=[
                    (1.0 - applicant_pd) * loan_gain,
                    -applicant_pd * loan_loss,
                    expected_value,
                ],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "#ef4444"}},
                increasing={"marker": {"color": "#10b981"}},
                totals={"marker": {"color": "#38bdf8" if is_approved else "#f43f5e"}},
            )
        )
        fig_waterfall.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            height=280,
            margin=dict(l=20, r=20, t=20, b=10),
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

    # Interpretable Decision Factors Table
    st.markdown("#### Factores Clave de Riesgo y Justificación")
    factors = [
        {
            "Variable": "Debt-to-Income (DTI)",
            "Valor": f"{u_dti * 100:.1f}%",
            "Impacto": "FAVORABLE" if u_dti < 0.35 else "DESFAVORABLE",
            "Explicación": "Endeudamiento saludable (<35%)"
            if u_dti < 0.35
            else "Sobreendeudamiento elevado (>35%)",
        },
        {
            "Variable": "Utilización Rotativa",
            "Valor": f"{u_util * 100:.1f}%",
            "Impacto": "FAVORABLE"
            if u_util < 0.30
            else ("NEUTRO" if u_util < 0.60 else "DESFAVORABLE"),
            "Explicación": "Uso conservador de tarjetas"
            if u_util < 0.30
            else "Líneas de crédito saturadas",
        },
        {
            "Variable": "Moras Históricas",
            "Valor": f"{u_delinq} eventos",
            "Impacto": "FAVORABLE" if u_delinq == 0 else "DESFAVORABLE",
            "Explicación": "Historial crediticio limpio"
            if u_delinq == 0
            else "Antecedentes de morosidad 30+ días",
        },
        {
            "Variable": "Capacidad de Pago (Monto / Ingreso)",
            "Valor": f"{u_amount / u_income:.1f}x ingresos",
            "Impacto": "FAVORABLE" if (u_amount / u_income) < 3.5 else "DESFAVORABLE",
            "Explicación": "Monto acorde a capacidad de ingresos"
            if (u_amount / u_income) < 3.5
            else "Apalancamiento alto respecto a ingreso mensual",
        },
    ]

    st.dataframe(factors, use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 4: Stress Testing Macroeconómico
# ==============================================================================

with tab_stress:
    st.markdown("### 🌪️ Stress Testing y Simulación de Escenarios Macroeconómicos")
    st.write(
        "Compruebe la resiliencia de la política frente a shocks exógenos adversos. "
        "El motor de decisión recalibra los umbrales para proteger la solvencia del capital."
    )

    scenario = st.selectbox(
        "Seleccionar Escenario de Stress:",
        [
            "🟢 Escenario Base (Economía Estable)",
            "🟡 Desaceleración Económica (+20% PD, +150 bps CoF)",
            "🔴 Shock Inflacionario & Tasas Banco Central (+450 bps CoF, +10% LGD)",
            "⚡ Crisis Severa / Recesión Profunda (+50% PD, +25% LGD, +600 bps CoF)",
        ],
        index=0,
    )

    # Apply shock multipliers
    if "Desaceleración" in scenario:
        stress_pd_mult = 1.20
        stress_cof = cost_of_funds + 0.015
        stress_lgd = lgd
    elif "Shock Inflacionario" in scenario:
        stress_pd_mult = 1.10
        stress_cof = cost_of_funds + 0.045
        stress_lgd = min(lgd + 0.10, 0.95)
    elif "Crisis Severa" in scenario:
        stress_pd_mult = 1.50
        stress_cof = cost_of_funds + 0.060
        stress_lgd = min(lgd + 0.25, 0.99)
    else:
        stress_pd_mult = 1.00
        stress_cof = cost_of_funds
        stress_lgd = lgd

    # Stressed portfolio with pd column
    df_stressed = df_portfolio.with_columns(
        pl.col("calibrated_pd").mul(stress_pd_mult).clip(0.001, 0.999).alias("pd")
    )

    stress_optimizer = CreditPolicyOptimizer(
        portfolio=df_stressed,
        params=EconomicParameters(
            default_lgd=stress_lgd,
            default_cost_of_funds=stress_cof,
            default_interest_rate=interest_rate,
            default_loan_term_months=loan_term,
        ),
    )

    stress_result = stress_optimizer.optimize_threshold(grid_size=300)

    delta_threshold = (
        stress_result.optimal_threshold - opt_result.optimal_threshold
    ) * 100
    delta_approval = (
        stress_result.optimal_policy.approval_rate - opt_result.optimal_policy.approval_rate
    ) * 100
    delta_stress_pnl = (
        stress_result.optimal_policy.expected_pnl - opt_result.optimal_policy.expected_pnl
    )

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric(
            "Umbral Óptimo Ajustado (p*)",
            f"{stress_result.optimal_threshold * 100:.2f}%",
            delta=f"{delta_threshold:.2f}%",
            delta_color="normal",
        )
    with col_s2:
        st.metric(
            "Tasa de Aprobación Reajustada",
            f"{stress_result.optimal_policy.approval_rate * 100:.1f}%",
            delta=f"{delta_approval:.1f}%",
            delta_color="inverse",
        )
    with col_s3:
        st.metric(
            "P&L Esperado Bajo Shock",
            f"${stress_result.optimal_policy.expected_pnl:,.0f}",
            delta=f"${delta_stress_pnl:,.0f}",
            delta_color="normal",
        )

    contract_pct = abs(delta_approval)
    st.info(
        f"**Diagnóstico de Riesgo:** Bajo el escenario '{scenario}', la política óptima "
        f"restringe la aprobación a **{stress_result.optimal_policy.approval_rate * 100:.1f}%** "
        f"(contracción de **{contract_pct:.1f}%**). "
        "Esto evita que el incremento en mora erosione el patrimonio institucional.",
        icon="🛡️",
    )


# ==============================================================================
# TAB 5: Arquitectura & API Playground
# ==============================================================================

with tab_api:
    st.markdown("### 🔌 Arquitectura de Producción y Conexión API")
    st.write(
        "Esta interfaz visual se respalda en una arquitectura robusta de microservicio en "
        "producción. A continuación se detalla cómo interactúa con los endpoints REST de "
        "**FastAPI**."
    )

    st.markdown(
        """
        ```mermaid
        graph LR
            UI["Streamlit Cockpit"] -->|"HTTP POST /decision"| API["FastAPI Endpoint"]
            API -->|"Valida Pydantic v2"| V["Schema Validator"]
            V -->|"Pipeline Scikit-Learn"| M["LightGBM + Calibración"]
            M -->|"Probabilidad PD"| E["Motor CreditPolicyOptimizer"]
            E -->|"JSON DecisionResponse"| UI
        ```
        """
    )

    st.markdown("#### Ejemplo de Petición cURL al Microservicio")
    curl_sample = """curl -X 'POST' \\
  'http://localhost:8000/decision' \\
  -H 'accept: application/json' \\
  -H 'Content-Type: application/json' \\
  -d '{
    "application_id": "APP-2026-001",
    "monthly_income": 4500.0,
    "debt_to_income": 0.28,
    "historical_delinquencies": 0,
    "revolving_utilization": 0.35,
    "loan_amount": 10000.0,
    "loan_term_months": 12,
    "interest_rate": 0.18,
    "cost_of_funds": 0.05,
    "lgd": 0.45
  }'"""
    st.code(curl_sample, language="bash")

    st.markdown("#### Inspección de Salud del Microservicio")
    if st.button("Probar Conexión con `http://localhost:8000/health`"):
        try:
            import httpx

            resp = httpx.get("http://localhost:8000/health", timeout=3.0)
            if resp.status_code == 200:
                st.success(f"API Online (HTTP 200): {resp.json()}")
            else:
                st.warning(f"API respondió con código {resp.status_code}")
        except Exception as exc:
            st.error(
                f"No se pudo conectar a la API local (`http://localhost:8000`). "
                f"Asegúrese de ejecutar `make run-api` en una terminal paralela. Detalle: {exc}"
            )
