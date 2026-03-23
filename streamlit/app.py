"""
Desafio Lighthouse — LH Nautical
Dashboard de Análise de Dados 2023–2024
Apresentado por Indicium Academy
"""

import os
import json
import warnings
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASETS = os.path.join(ROOT, "datasets")
OUTPUTS  = os.path.join(ROOT, "outputs")
ASSETS   = os.path.join(ROOT, "assets")

# ── Brand colors ───────────────────────────────────────────────────────────
BG       = "#111118"
CARD     = "#1C1C28"
BORDER   = "#2BB8CC33"
TEAL     = "#2BB8CC"
GREEN    = "#00D166"
BLUE     = "#3A8FD1"
ORANGE   = "#E09045"
RED      = "#E05252"
WHITE    = "#FFFFFF"
GRAY     = "#8892A4"
GRAY2    = "#B0BAC8"

# ── SVG icons (reusable, no emojis) ────────────────────────────────────────
def _svg(path_d, color, w=16, h=16, extra=""):
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round">{path_d}</svg>'
    )

ICO_OK   = _svg('<polyline points="20 6 9 17 4 12"/>', "#00D166")
ICO_WARN = _svg('<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>', "#E09045")
ICO_ERR  = _svg('<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>', "#E05252")
ICO_INFO = _svg('<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>', "#2BB8CC")

def card_header(icon_svg, label, color):
    """Renders a card section header with SVG icon — no emoji, no anchor link."""
    return (
        f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:14px'>"
        f"{icon_svg}"
        f"<span style='font-size:0.95rem;font-weight:600;color:{color}'>{label}</span>"
        f"</div>"
    )

# ── Plotly base layout ─────────────────────────────────────────────────────
def plotly_layout(**kwargs):
    base = dict(
        paper_bgcolor=CARD,
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=WHITE, family="Inter, sans-serif", size=13),
        title=dict(font=dict(color=WHITE, size=15), x=0.01),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", showline=False, zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", showline=False, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=GRAY2)),
        margin=dict(l=16, r=16, t=48, b=16),
    )
    base.update(kwargs)
    return base


# ── CSS ────────────────────────────────────────────────────────────────────
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.stApp {{ background-color: {BG}; color: {WHITE}; }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: {CARD};
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] * {{ color: {GRAY2} !important; }}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{ color: {WHITE} !important; }}

/* Radio nav */
[data-testid="stSidebar"] [data-testid="stRadio"] label {{
    padding: 6px 12px;
    border-radius: 8px;
    cursor: pointer;
    display: block;
    transition: background 0.15s;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
    background: rgba(43,184,204,0.12);
    color: {TEAL} !important;
}}

/* Metrics */
[data-testid="metric-container"] {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 16px 20px;
}}
[data-testid="stMetricValue"] {{
    color: {TEAL} !important;
    font-size: 1.55rem !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricLabel"] {{
    color: {GRAY} !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
[data-testid="stMetricDelta"] {{ font-size: 0.8rem !important; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: {CARD};
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {GRAY};
    border-radius: 7px;
    padding: 6px 18px;
    border: none;
    font-weight: 500;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {TEAL}28, {GREEN}28) !important;
    color: {TEAL} !important;
    font-weight: 600;
}}

/* Dataframes */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    overflow: hidden;
}}
[data-testid="stDataFrame"] th {{
    background: {CARD} !important;
    color: {TEAL} !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

/* Alerts */
[data-testid="stInfo"] {{ background: rgba(43,184,204,0.08); border-color: {TEAL}; }}
[data-testid="stWarning"] {{ background: rgba(224,144,69,0.08); border-color: {ORANGE}; }}

/* Dividers */
hr {{ border-color: rgba(255,255,255,0.07) !important; }}

/* Hide Streamlit branding */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ visibility: hidden; }}

/* Prevent sidebar from collapsing — navigation would be lost */
[data-testid="stSidebarCollapseButton"] {{ display: none !important; }}
section[data-testid="stSidebar"] {{ transform: none !important; min-width: 176px !important; }}

/* Custom components */
.ind-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}}
.ind-card h4 {{ margin-top: 0; }}

.section-title {{
    border-left: 4px solid {TEAL};
    padding-left: 14px;
    margin-bottom: 8px;
}}
.section-title h2 {{ margin: 0; color: {WHITE}; font-size: 1.5rem; font-weight: 700; }}
.section-title p  {{ margin: 4px 0 0 0; color: {GRAY}; font-size: 0.9rem; }}

.answer-box {{
    background: rgba(43,184,204,0.07);
    border: 1px solid rgba(43,184,204,0.3);
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
    color: {GRAY2};
}}
.answer-box strong {{ color: {TEAL}; }}

/* ── Flip cards ── */
.flip-card-wrap {{
    perspective: 1000px;
    cursor: pointer;
    height: 240px;
    user-select: none;
}}
.flip-card-inner {{
    position: relative;
    width: 100%;
    height: 100%;
    transition: transform 0.55s cubic-bezier(0.4, 0, 0.2, 1);
    transform-style: preserve-3d;
}}
.flip-card-wrap.flipped .flip-card-inner {{
    transform: rotateY(180deg);
}}
/* CSS-only flip via checkbox hack */
#flip-gabriel:checked + .flip-card-wrap .flip-card-inner,
#flip-marina:checked  + .flip-card-wrap .flip-card-inner,
#flip-almir:checked   + .flip-card-wrap .flip-card-inner {{
    transform: rotateY(180deg);
}}
.flip-card-wrap {{
    display: block;
    cursor: pointer;
}}
.flip-front, .flip-back {{
    position: absolute;
    width: 100%;
    height: 100%;
    backface-visibility: hidden;
    -webkit-backface-visibility: hidden;
    border-radius: 14px;
    border: 1px solid {BORDER};
    background: {CARD};
    box-sizing: border-box;
}}
.flip-front {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 22px;
    text-align: center;
}}
.flip-front h4 {{ margin: 10px 0 4px; font-size: 1rem; font-weight: 600; }}
.flip-front .role {{ font-size: 0.78rem; color: {GRAY}; margin: 0; }}
.flip-hint {{
    font-size: 0.72rem;
    color: {GRAY};
    margin-top: 12px;
    display: flex;
    align-items: center;
    gap: 5px;
    justify-content: center;
    opacity: 0.7;
}}
.flip-back {{
    transform: rotateY(180deg);
    padding: 20px 22px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    justify-content: center;
}}
.flip-back h5 {{ margin: 0 0 10px; font-size: 0.9rem; font-weight: 600; }}
.flip-back ul {{ margin: 0; padding-left: 16px; }}
.flip-back ul li {{
    font-size: 0.79rem;
    color: {GRAY2};
    line-height: 1.55;
    margin-bottom: 5px;
}}
.flip-back ul li strong {{ color: {WHITE}; }}

/* Navigation guide cards */
.nav-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
    display: flex;
    align-items: flex-start;
    gap: 14px;
    transition: border-color 0.2s;
}}
.nav-card:hover {{ border-color: {TEAL}; }}
.nav-card .nav-icon {{ flex-shrink: 0; opacity: 0.85; }}
.nav-card .nav-text h5 {{ margin: 0 0 4px 0; font-size: 0.95rem; font-weight: 600; color: {WHITE}; }}
.nav-card .nav-text p  {{ margin: 0; font-size: 0.83rem; color: {GRAY}; line-height: 1.5; }}

/* Stakeholder summary toggle */
.stakeholder-summary {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px 24px;
    margin-top: 12px;
}}
.stakeholder-summary h5 {{ margin: 0 0 12px 0; font-size: 0.95rem; font-weight: 600; }}
.stakeholder-summary ul {{ margin: 0; padding-left: 18px; color: {GRAY2}; font-size: 0.87rem; line-height: 1.8; }}

/* Sidebar nav HTML links */
.ind-nav-link {{
    display: flex;
    align-items: center;
    justify-content: flex-start;
    gap: 10px;
    padding: 6px 10px;
    border-radius: 8px;
    text-decoration: none !important;
    color: #B0BAC8;
    font-size: 0.84rem;
    text-align: left;
    transition: all 0.15s;
    margin-bottom: 1px;
    cursor: pointer;
    width: 100%;
    box-sizing: border-box;
}}
.ind-nav-link:hover {{
    background: rgba(43,184,204,0.12);
    color: #2BB8CC !important;
}}
.ind-nav-link.active {{
    background: rgba(43,184,204,0.18);
    color: #2BB8CC !important;
    font-weight: 600;
}}
.ind-nav-link svg {{ flex-shrink: 0; }}
.ind-nav-link span {{ flex: 1; }}

/* Diagnostico item icons */
.diag-item {{ display: flex; align-items: flex-start; gap: 10px; margin-bottom: 10px; color: #B0BAC8; font-size: 0.9rem; line-height: 1.5; }}
.diag-item svg {{ flex-shrink: 0; margin-top: 2px; }}

/* KPI grid */
.kpi-grid {{ display: grid; grid-template-columns: repeat(4,1fr); gap: 1px; background: rgba(43,184,204,0.15); border-radius: 12px; overflow: hidden; margin-bottom: 16px; }}
.kpi-cell {{ background: #1C1C28; padding: 16px 20px; }}
.kpi-label {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em; color: #8892A4; margin-bottom: 6px; }}
.kpi-value {{ font-size: 1.45rem; font-weight: 700; color: #2BB8CC; }}
.kpi-grid-3 {{ grid-template-columns: repeat(3,1fr); }}
</style>
"""

# ─────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Desafio Lighthouse — LH Nautical",
    page_icon=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "assets", "indicium-200x73.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# DATA LOADERS (cached)
# ─────────────────────────────────────────────────────────────────────────

def _parse_date(s):
    for fmt in ["%Y-%m-%d", "%d-%m-%Y"]:
        try:
            return datetime.strptime(str(s), fmt)
        except ValueError:
            pass
    return None


@st.cache_data(show_spinner=False)
def load_vendas() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATASETS, "vendas_2023_2024.csv"))
    df["sale_date"] = pd.to_datetime(df["sale_date"].apply(_parse_date))
    return df.dropna(subset=["sale_date"])


@st.cache_data(show_spinner=False)
def load_produtos():
    df_raw = pd.read_csv(os.path.join(DATASETS, "produtos_raw.csv"))

    def norm_cat(c):
        c2 = unicodedata.normalize("NFD", str(c).lower())
        c2 = "".join(ch for ch in c2 if unicodedata.category(ch) != "Mn")
        c2 = c2.replace(" ", "")
        if "eletron" in c2 or "eletrun" in c2:
            return "eletrônicos"
        elif "propul" in c2 or c2.startswith("prop"):
            return "propulsão"
        elif "ancor" in c2 or "encor" in c2:
            return "ancoragem"
        return c

    # Mirrors exercicio_2/script.py exactly: normalize in-place, drop_duplicates on all cols
    df = df_raw.copy()
    df["actual_category"] = df["actual_category"].apply(norm_cat)
    df["price"] = (
        df["price"]
        .str.replace("R$", "", regex=False)
        .str.strip()
        .astype(float)
    )
    df_clean = df.drop_duplicates().copy()
    # Aliases for display
    df_clean["categoria"] = df_clean["actual_category"]
    df_clean["price_num"] = df_clean["price"]
    return df_raw, df_clean


@st.cache_data(show_spinner=False)
def load_custos() -> pd.DataFrame:
    with open(os.path.join(DATASETS, "custos_importacao.json"), encoding="utf-8") as f:
        dados = json.load(f)
    rows = []
    for p in dados:
        for h in p["historic_data"]:
            rows.append({
                "id_product":   p["product_id"],
                "product_name": p["product_name"],
                "category":     p["category"],
                "start_date":   datetime.strptime(h["start_date"], "%d/%m/%Y"),
                "usd_price":    h["usd_price"],
            })
    return pd.DataFrame(rows).sort_values("start_date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_analise_prejuizo():
    path = os.path.join(OUTPUTS, "analise_prejuizo.csv")
    if os.path.exists(path):
        return pd.read_csv(path)

    # Compute on-the-fly from raw datasets
    df_vendas = load_vendas()
    df_custos = load_custos()

    # Merge-based approach: for each sale find the last USD price before sale_date
    # Use merge_asof (sorted merge on date, per product)
    df_v = df_vendas[["id_product", "qtd", "total", "sale_date"]].copy()
    df_c = df_custos[["id_product", "start_date", "usd_price"]].copy()

    # merge_asof requer left globalmente ordenado por left_on — processa por produto
    TAXA_CAMBIO = 5.0
    partes = []
    for pid, vendas_prod in df_v.groupby("id_product"):
        custos_prod = df_c[df_c["id_product"] == pid][["start_date", "usd_price"]].sort_values("start_date")
        if custos_prod.empty:
            continue
        v_sorted = vendas_prod.sort_values("sale_date")
        m = pd.merge_asof(v_sorted, custos_prod, left_on="sale_date", right_on="start_date", direction="backward")
        partes.append(m)

    merged = pd.concat(partes, ignore_index=True)
    merged["custo_brl"] = merged["usd_price"] * merged["qtd"] * TAXA_CAMBIO

    df_result = (
        merged.dropna(subset=["usd_price"])
        .groupby("id_product")
        .agg(custo_total=("custo_brl", "sum"), receita_total=("total", "sum"))
        .reset_index()
    )
    df_result["prejuizo_total"] = (df_result["custo_total"] - df_result["receita_total"]).clip(lower=0)
    df_result["percentual_perda"] = df_result["prejuizo_total"] / df_result["custo_total"].replace(0, float("nan"))
    df_result = df_result[df_result["prejuizo_total"] > 0].sort_values("prejuizo_total", ascending=False)
    return df_result


@st.cache_data(show_spinner=False)
def load_clientes() -> pd.DataFrame:
    with open(os.path.join(DATASETS, "clientes_crm.json"), encoding="utf-8") as f:
        dados = json.load(f)
    return pd.DataFrame(dados)


# ─────────────────────────────────────────────────────────────────────────
# NAVIGATION (query params)
# ─────────────────────────────────────────────────────────────────────────
page = st.query_params.get("p", "home")

NAV_ICONS = {
    "home": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>',
    "q1":   '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
    "q2":   '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/></svg>',
    "q3":   '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>',
    "q4":   '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>',
    "q5":   '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
    "q6":   '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>',
    "q7":   '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "q8":   '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>',
}

NAV = [
    ("home", "Início"),
    ("q1",   "Qualidade dos Dados"),
    ("q2",   "Catálogo de Produtos"),
    ("q3",   "Custos de Importação"),
    ("q4",   "Análise de Prejuízo"),
    ("q5",   "Clientes Fiéis"),
    ("q6",   "Calendário de Vendas"),
    ("q7",   "Previsão de Demanda"),
    ("q8",   "Recomendação"),
]

# ─────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    logo = os.path.join(ASSETS, "indicium-200x73.png")
    if os.path.exists(logo):
        st.image(logo, width=155)
    st.markdown("---")
    st.markdown(
        f"<p style='color:{GRAY}; font-size:0.75rem; text-transform:uppercase; "
        f"letter-spacing:0.08em; margin:0 0 8px 0'>Navegação</p>",
        unsafe_allow_html=True,
    )

    nav_html = ""
    for pid, label in NAV:
        active_cls = "active" if page == pid else ""
        icon = NAV_ICONS.get(pid, "")
        nav_html += (
            f'<a href="?p={pid}" class="ind-nav-link {active_cls}" target="_self">'
            f'{icon}<span>{label}</span></a>'
        )
    st.markdown(nav_html, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        f"<p style='color:{GRAY}; font-size:0.78rem; line-height:1.7'>"
        f"<strong style='color:{WHITE}'>Cecília Zica Camargo</strong><br>"
        f"<span style='color:{TEAL}'>Sistemas de Informação</span><br>"
        f"Especialização em Dados · Março 2026</p>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────
def section(title: str, subtitle: str = ""):
    sub_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"<div class='section-title'><h2>{title}</h2>{sub_html}</div>",
        unsafe_allow_html=True,
    )


def answer_box(html: str):
    st.markdown(f"<div class='answer-box'>{html}</div>", unsafe_allow_html=True)


def ind_card(html: str):
    st.markdown(f"<div class='ind-card'>{html}</div>", unsafe_allow_html=True)


def fmt_brl(v: float) -> str:
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ─────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────
if page == "home":
    # Header with Indicium logo
    logo_path = os.path.join(ASSETS, "logo_indicium-tecnologia-de-dados-ltda_bRcAaV.png")
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.markdown("<div style='padding-top:18px'>", unsafe_allow_html=True)
        if os.path.exists(logo_path):
            st.image(logo_path, width=140)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_title:
        st.markdown(
            f"""<div style="padding: 18px 0 10px 0;">
                <h1 style="color:{TEAL}; font-size:2.1rem; font-weight:700; margin:0 0 4px 0">
                    Desafio Lighthouse
                </h1>
                <p style="color:{GRAY}; font-size:0.95rem; margin:0">
                    LH Nautical — Relatório de Análise de Dados 2023–2024
                </p>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Stakeholders — sem descrição, com ícones SVG
    st.markdown(
        f"<h3 style='color:{WHITE}; margin-bottom:16px; font-size:1.1rem; "
        f"text-transform:uppercase; letter-spacing:0.06em'>Destinatários</h3>",
        unsafe_allow_html=True,
    )

    SVG_CODE = (
        '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" '
        'stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">'
        '<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>'
        '</svg>'
    )
    SVG_CHART = (
        '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" '
        'stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">'
        '<line x1="18" y1="20" x2="18" y2="10"/>'
        '<line x1="12" y1="20" x2="12" y2="4"/>'
        '<line x1="6" y1="20" x2="6" y2="14"/>'
        '</svg>'
    )
    SVG_BUILDING = (
        '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" '
        'stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'
        '<polyline points="9 22 9 12 15 12 15 22"/>'
        '</svg>'
    )

    FLIP_ICON = (
        '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M23 4v6h-6"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>'
        '</svg>'
    )
    st.markdown(
        f"""<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:16px;">

  <!-- Gabriel Santos -->
  <div>
    <input type="checkbox" id="flip-gabriel" style="display:none">
    <label for="flip-gabriel" class="flip-card-wrap">
      <div class="flip-card-inner">
        <div class="flip-front">
          <div>{SVG_CODE}</div>
          <h4 style="color:{TEAL}">Gabriel Santos</h4>
          <p class="role">Tech Lead</p>
          <div class="flip-hint">{FLIP_ICON}&nbsp;Clique para ver o perfil</div>
        </div>
        <div class="flip-back">
          <h5 style="color:{TEAL}">Gabriel Santos — Tech Lead</h5>
          <ul>
            <li><strong>Qualidade dos Dados:</strong> Dataset íntegro, mas datas em dois formatos — exige tratamento antes de análises temporais.</li>
            <li><strong>Catálogo de Produtos:</strong> 157 → 150 únicos. 7 duplicatas removidas. Categorias padronizadas.</li>
            <li><strong>Custos de Importação:</strong> Pipeline USD→BRL com câmbio histórico real e cache em CSV.</li>
            <li><strong>Previsão de Demanda:</strong> Média móvel 7 dias com avaliação por MAE, MAPE e RMSE.</li>
            <li><strong>Recomendação:</strong> Sistema "quem comprou isso também levou" via similaridade de cosseno.</li>
          </ul>
        </div>
      </div>
    </label>
  </div>

  <!-- Marina Costa -->
  <div>
    <input type="checkbox" id="flip-marina" style="display:none">
    <label for="flip-marina" class="flip-card-wrap">
      <div class="flip-card-inner">
        <div class="flip-front">
          <div>{SVG_CHART}</div>
          <h4 style="color:{GREEN}">Marina Costa</h4>
          <p class="role">Gestora de Negócios</p>
          <div class="flip-hint">{FLIP_ICON}&nbsp;Clique para ver o perfil</div>
        </div>
        <div class="flip-back">
          <h5 style="color:{GREEN}">Marina Costa — Gestora de Negócios</h5>
          <ul>
            <li><strong>Análise de Prejuízo:</strong> Produto 72 com R$ 36,2 M de prejuízo — 36% de margem negativa.</li>
            <li><strong>Custos de Importação:</strong> Variação cambial impacta margens. Propulsão tem maior exposição ao dólar.</li>
            <li><strong>Clientes Fiéis:</strong> Cliente 47 lidera com ticket médio de R$ 336 k.</li>
            <li><strong>Calendário:</strong> Domingo é o pior dia — oportunidade de campanhas no fim de semana.</li>
          </ul>
        </div>
      </div>
    </label>
  </div>

  <!-- Sr. Almir -->
  <div>
    <input type="checkbox" id="flip-almir" style="display:none">
    <label for="flip-almir" class="flip-card-wrap">
      <div class="flip-card-inner">
        <div class="flip-front">
          <div>{SVG_BUILDING}</div>
          <h4 style="color:{ORANGE}">Sr. Almir</h4>
          <p class="role">Fundador</p>
          <div class="flip-hint">{FLIP_ICON}&nbsp;Clique para ver o perfil</div>
        </div>
        <div class="flip-back">
          <h5 style="color:{ORANGE}">Sr. Almir — Fundador</h5>
          <ul>
            <li><strong>Visão Geral:</strong> 9.895 transações, 150 produtos, 49 clientes no período 2023–2024.</li>
            <li><strong>Maior Problema:</strong> Produto 72 com R$ 36,2 M de prejuízo — revisão de precificação urgente.</li>
            <li><strong>Melhor Cliente:</strong> Cliente 47, R$ 336 k de ticket médio — candidato a programa de fidelidade.</li>
            <li><strong>Oportunidade:</strong> GPS Garmin com 87% de similaridade ao Produto 94 — base para cross-sell.</li>
          </ul>
        </div>
      </div>
    </label>
  </div>

</div>""",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Executive summary
    st.markdown(
        f"<h3 style='color:{WHITE}; margin-bottom:16px; font-size:1.1rem; "
        f"text-transform:uppercase; letter-spacing:0.06em'>Resumo Executivo</h3>",
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    cols[0].metric("Transações Analisadas", "9.895")
    cols[1].metric("Período", "2023–2024")
    cols[2].metric("Produtos Únicos", "150")
    cols[3].metric("Clientes Ativos", "49")

    cols2 = st.columns(4)
    cols2[0].metric("Maior Prejuízo", "R$ 36,2 M", delta="Produto 72 · 36% de perda", delta_color="inverse")
    cols2[1].metric("Melhor Ticket Médio", "R$ 336 k", delta="Cliente 47 · Top fiel")
    cols2[2].metric("Pior Dia de Vendas", "Domingo", delta="Média: R$ 3,3 M / dia")
    cols2[3].metric("Melhor Similaridade", "0,8696", delta="GPS Garmin → Produto 94")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Navegação rápida ─────────────────────────────────────────────────
    st.markdown(
        f"<h3 style='color:{WHITE}; margin-bottom:16px; font-size:1.1rem; "
        f"text-transform:uppercase; letter-spacing:0.06em'>Guia de Páginas</h3>",
        unsafe_allow_html=True,
    )

    NAV_GUIDE = [
        ("q1", "Qualidade dos Dados",
         "Análise exploratória do dataset de vendas 2023–2024: distribuição, outliers e diagnóstico de qualidade.",
         BLUE,
         '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>'),
        ("q2", "Catálogo de Produtos",
         "Normalização de categorias, tratamento de preços e remoção de duplicatas no catálogo.",
         GREEN,
         '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/></svg>'),
        ("q3", "Custos de Importação",
         "Normalização de custos em USD com câmbio real para BRL por período histórico.",
         TEAL,
         '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>'),
        ("q4", "Análise de Prejuízo",
         "Identificação dos produtos com maior perda financeira e percentual de margem negativa.",
         RED,
         '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/></svg>'),
        ("q5", "Clientes Fiéis",
         "Ranking de clientes por volume, frequência e ticket médio. Identificação dos top compradores.",
         ORANGE,
         '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>'),
        ("q6", "Calendário de Vendas",
         "Análise de desempenho por dia da semana. Identificação dos melhores e piores dias para vendas.",
         BLUE,
         '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>'),
        ("q7", "Previsão de Demanda",
         "Projeção de vendas com média móvel de 7 dias e avaliação do erro (MAE) do modelo.",
         GREEN,
         '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>'),
        ("q8", "Recomendação de Produtos",
         "Sistema 'Quem comprou isso também levou…' — encontra produtos frequentemente comprados juntos.",
         TEAL,
         '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>'),
    ]

    col_a, col_b = st.columns(2)
    for i, (pid, title, desc, color, icon_svg) in enumerate(NAV_GUIDE):
        col = col_a if i % 2 == 0 else col_b
        with col:
            st.markdown(
                f"""<div class='nav-card'>
                    <div class='nav-icon' style='color:{color}'>{icon_svg}</div>
                    <div class='nav-text'>
                        <h5>{title}</h5>
                        <p>{desc}</p>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
            if st.button(f"Acessar: {title}", key=f"home_nav_{pid}", use_container_width=True):
                st.query_params["p"] = pid
                st.rerun()



# ─────────────────────────────────────────────────────────────────────────
# PAGE: Q1 — EDA
# ─────────────────────────────────────────────────────────────────────────
elif page == "q1":
    section("Qualidade dos Dados", "Diagnóstico do dataset de vendas 2023–2024")

    with st.spinner("Carregando dados…"):
        df = load_vendas()

    # KPI grid — custom HTML for visual richness
    vmin = df["total"].min()
    vmax = df["total"].max()
    vmean = df["total"].mean()
    dmin = df["sale_date"].min().strftime("%d/%m/%Y")
    dmax = df["sale_date"].max().strftime("%d/%m/%Y")
    total_nulos = int(df.isnull().sum().sum())

    st.markdown(
        f"""<div class='kpi-grid'>
            <div class='kpi-cell'>
                <div class='kpi-label'>Total de Linhas</div>
                <div class='kpi-value'>{f"{len(df):,}".replace(",", ".")}</div>
            </div>
            <div class='kpi-cell'>
                <div class='kpi-label'>Total de Colunas</div>
                <div class='kpi-value'>{df.shape[1]}</div>
            </div>
            <div class='kpi-cell'>
                <div class='kpi-label'>Data Mínima</div>
                <div class='kpi-value'>{dmin}</div>
            </div>
            <div class='kpi-cell'>
                <div class='kpi-label'>Data Máxima</div>
                <div class='kpi-value'>{dmax}</div>
            </div>
        </div>
        <div class='kpi-grid' style='margin-bottom:24px'>
            <div class='kpi-cell'>
                <div class='kpi-label'>Valor Mínimo</div>
                <div class='kpi-value'>{fmt_brl(vmin)}</div>
            </div>
            <div class='kpi-cell'>
                <div class='kpi-label'>Valor Máximo</div>
                <div class='kpi-value'>{fmt_brl(vmax)}</div>
            </div>
            <div class='kpi-cell'>
                <div class='kpi-label'>Valor Médio</div>
                <div class='kpi-value'>{fmt_brl(vmean)}</div>
            </div>
            <div class='kpi-cell'>
                <div class='kpi-label'>Valores Nulos</div>
                <div class='kpi-value' style='color:{"#00D166" if total_nulos == 0 else RED}'>{total_nulos}</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["Distribuição de Valores", "Receita por Mês", "Amostra dos Dados"])

    with tab1:
        vals = df["total"].dropna().values
        Q1_v = df["total"].quantile(0.25)
        Q3_v = df["total"].quantile(0.75)
        IQR_v = Q3_v - Q1_v
        upper_fence = Q3_v + 1.5 * IQR_v
        n_outliers = int((df["total"] > upper_fence).sum())

        col_dist, col_sc = st.columns(2)

        with col_dist:
            lim_k = int(upper_fence / 1000)
            faixas = [
                ("Até R$ 1k",                    (0,            1_000)),
                ("R$ 1k – 10k",                  (1_000,        10_000)),
                ("R$ 10k – 100k",                (10_000,       100_000)),
                ("R$ 100k – 500k",               (100_000,      500_000)),
                (f"R$ 500k – {lim_k}k*",         (500_000,      upper_fence)),
                (f"R$ {lim_k}k – 1M  ⚠",        (upper_fence,  1_000_000)),
                ("R$ 1M – 1,5M  ⚠",              (1_000_000,    1_500_000)),
                ("Acima R$ 1,5M  ⚠",             (1_500_000,    float("inf"))),
            ]
            df_faixas = pd.DataFrame([
                {
                    "Faixa": lbl,
                    "Transações": int(((vals >= lo) & (vals < hi)).sum()),
                    "Outlier": "⚠" in lbl,
                }
                for lbl, (lo, hi) in faixas
            ])
            fig = px.bar(
                df_faixas, x="Faixa", y="Transações",
                title="Transações por Faixa de Valor",
                color="Outlier",
                color_discrete_map={False: TEAL, True: RED},
                text="Transações",
            )
            fig.update_traces(textposition="outside", textfont_size=11)
            fig.update_layout(**plotly_layout(), showlegend=False)
            fig.add_vline(
                x=4.5, line_dash="dash", line_color=ORANGE, line_width=1.5,
                annotation_text=f"Limite IQR: {fmt_brl(upper_fence)}",
                annotation_position="top",
                annotation_font_color=ORANGE,
                annotation_font_size=11,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f"""<div style='background:{CARD}; border:1px solid {BORDER};
                    border-left:3px solid {BLUE}; border-radius:8px;
                    padding:12px 16px; margin-top:-8px; font-size:0.82rem; color:{GRAY2}; line-height:1.6'>
                    <strong style='color:{WHITE}'>Como foi calculado o limite de {fmt_brl(upper_fence)}?</strong><br>
                    Usamos o método IQR: pegamos o valor que separa os 75% mais baixos das vendas (Q3)
                    e subtraímos o valor dos 25% mais baixos (Q1). Multiplicamos essa diferença por 1,5
                    e somamos ao Q3. Qualquer venda acima disso é considerada <strong style='color:{RED}'>fora do padrão</strong>
                    — pode ser uma venda legítima de alto valor, mas merece investigação.
                </div>""",
                unsafe_allow_html=True,
            )

        with col_sc:
            # Box plot — forma padrão e intuitiva de mostrar outliers
            fig2 = px.box(
                df, y="total",
                title="Onde estão os valores fora do padrão?",
                labels={"total": "Valor da Transação (R$)"},
                color_discrete_sequence=[TEAL],
                points="outliers",
            )
            fig2.update_traces(
                marker_color=RED,
                marker_size=4,
                marker_opacity=0.6,
                line_color=TEAL,
                fillcolor=BLUE,
            )
            fig2.add_hline(
                y=upper_fence, line_dash="dash", line_color=ORANGE, line_width=1.5,
                annotation_text=f"Limite normal: {fmt_brl(upper_fence)}",
                annotation_position="top right",
                annotation_font_color=ORANGE,
            )
            fig2.update_layout(**plotly_layout())
            st.plotly_chart(fig2, use_container_width=True)

        answer_box(
            f"<strong>{n_outliers} vendas fora do padrão</strong> (pontos vermelhos acima da linha laranja). "
            f"A caixa azul representa onde estão a maioria das transações — "
            f"qualquer ponto vermelho acima da linha merece investigação."
        )

        # Explicação para o fundador
        st.markdown(
            f"""<div style='background:{CARD}; border-left: 3px solid {ORANGE};
                border-radius:10px; padding:16px 20px; margin-top:8px;'>
                <div style='font-size:0.8rem; color:{ORANGE}; font-weight:600;
                    text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;'>
                    Para entender sem complicação
                </div>
                <p style='color:{GRAY2}; font-size:0.88rem; line-height:1.7; margin:0'>
                    Imagine que a maioria das suas vendas vale entre
                    <strong style='color:{WHITE}'>R$ 50 mil e R$ 800 mil</strong>.
                    Isso é o "normal" do negócio.<br>
                    Mas {n_outliers} vendas estão <strong style='color:{RED}'>muito acima disso</strong>
                    — chegando a R$ 2,2 milhões numa única transação.<br>
                    Vale checar: são vendas reais de clientes grandes, ou podem ser erros de digitação
                    no sistema? Esse tipo de verificação evita surpresas no balanço.
                </p>
            </div>""",
            unsafe_allow_html=True,
        )

    with tab2:
        import calendar
        MONTH_PT = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",
                    7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
        df_m = (
            df.assign(
                ano=df["sale_date"].dt.year,
                mes_num=df["sale_date"].dt.month,
            )
            .groupby(["ano","mes_num"])["total"].sum()
            .reset_index()
        )
        df_m["Mês"] = df_m.apply(
            lambda r: f"{MONTH_PT[r['mes_num']]}/{str(r['ano'])[2:]}", axis=1
        )
        df_m = df_m.sort_values(["ano","mes_num"]).reset_index(drop=True)
        df_m.rename(columns={"total":"Receita"}, inplace=True)

        fig2 = px.bar(
            df_m, x="Mês", y="Receita",
            title="Receita Mensal — LH Nautical (2023–2024)",
            labels={"Mês": "", "Receita": "Receita (R$)"},
            color="Receita",
            color_continuous_scale=[[0, BLUE], [1, TEAL]],
        )

        # Trend line (green, Indicium style)
        x_idx = np.arange(len(df_m))
        y_vals = df_m["Receita"].values
        z = np.polyfit(x_idx, y_vals, 1)
        trend = np.polyval(z, x_idx)
        fig2.add_scatter(
            x=df_m["Mês"], y=trend,
            mode="lines", line=dict(color=GREEN, width=2, dash="dot"),
            name="Tendência",
        )

        fig2.update_layout(**plotly_layout(), coloraxis_showscale=False)
        fig2.update_xaxes(
            tickmode="array",
            tickvals=df_m["Mês"].tolist(),
            ticktext=df_m["Mês"].tolist(),
            tickangle=45,
            tickfont=dict(size=11),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        COL_DESC = {
            "id": "id (identificador único da transação)",
            "id_client": "id_client (identificador do cliente)",
            "id_product": "id_product (identificador do produto)",
            "qtd": "qtd (quantidade de itens vendidos)",
            "total": "total (valor total da venda em R$)",
            "sale_date": "sale_date (data da venda)",
        }
        df_show = df.head(20).assign(sale_date=df["sale_date"].dt.strftime("%Y-%m-%d")).copy()
        df_show.columns = [COL_DESC.get(c, c) for c in df_show.columns]
        st.dataframe(df_show, use_container_width=True, hide_index=True)

    SVG_OK = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#00D166" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>'
    SVG_WARN = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#E09045" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
    SVG_ERR = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#E05252" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>'

    ind_card(
        f"""<div style='font-size:0.95rem; font-weight:600; color:{WHITE}; margin-bottom:14px'>Diagnóstico de Qualidade</div>
        <div class='diag-item'>{SVG_OK}<span><strong style='color:{WHITE}'>9.895 registros e 6 colunas</strong> — sem valores nulos</span></div>
        <div class='diag-item'>{SVG_WARN}<span><strong>Outliers:</strong> razão máx/mín ~7.500× requer investigação</span></div>
        <div class='diag-item'>{SVG_WARN}<span><strong>Datas mistas:</strong> <code>sale_date</code> usa <code>YYYY-MM-DD</code> e <code>DD-MM-YYYY</code> simultaneamente</span></div>
        <div class='diag-item'>{SVG_ERR}<span><strong>Conclusão:</strong> dataset <strong style='color:{RED}'>NÃO está pronto</strong> — exige padronização de datas e avaliação dos outliers</span></div>"""
    )


# ─────────────────────────────────────────────────────────────────────────
# PAGE: Q2 — Products
# ─────────────────────────────────────────────────────────────────────────
elif page == "q2":
    section("Catálogo de Produtos", "Padronização de categorias, preços e remoção de duplicatas")

    with st.spinner("Carregando produtos…"):
        df_raw, df_clean = load_produtos()

    c = st.columns(4)
    c[0].metric("Linhas Originais", str(len(df_raw)))
    c[1].metric("Duplicatas Removidas", str(len(df_raw) - len(df_clean)))
    c[2].metric("Linhas Finais", str(len(df_clean)))
    c[3].metric("Categorias", "3")

    st.markdown("---")

    tab1, tab2 = st.tabs(["Visualizações", "Dados Normalizados"])

    with tab1:
        col_a, col_b = st.columns(2)

        with col_a:
            cat_counts = df_clean["categoria"].value_counts().reset_index()
            cat_counts.columns = ["Categoria", "Quantidade"]
            fig = px.bar(
                cat_counts, x="Categoria", y="Quantidade",
                title="Produtos por Categoria",
                color="Categoria",
                color_discrete_sequence=[TEAL, GREEN, BLUE],
            )
            fig.update_layout(**plotly_layout(), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig2 = px.box(
                df_clean, x="categoria", y="price_num",
                title="Distribuição de Preços por Categoria (R$)",
                labels={"categoria": "Categoria", "price_num": "Preço (R$)"},
                color="categoria",
                color_discrete_sequence=[TEAL, GREEN, BLUE],
            )
            fig2.update_layout(**plotly_layout(), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.dataframe(
            df_clean[["name", "categoria", "price_num", "code"]]
            .rename(columns={"name": "Produto", "categoria": "Categoria",
                              "price_num": "Preço (R$)", "code": "Código"})
            .head(30),
            use_container_width=True, hide_index=True,
        )

    ind_card(
        card_header(ICO_OK, "Resultado", GREEN) +
        f"""<p style='color:{GRAY2}'>
            <strong style='color:{WHITE}'>7 duplicatas removidas</strong> (157 → 150 linhas).
            Categorias padronizadas para 3 grupos:
            <strong style='color:{TEAL}'>eletrônicos</strong>,
            <strong style='color:{GREEN}'>propulsão</strong> e
            <strong style='color:{BLUE}'>ancoragem</strong> — 50 produtos em cada.
        </p>"""
    )


# ─────────────────────────────────────────────────────────────────────────
# PAGE: Q3 — Import Costs
# ─────────────────────────────────────────────────────────────────────────
elif page == "q3":
    section("Custos de Importação", "Achatamento de JSON hierárquico → tabela flat")

    with st.spinner("Processando JSON de custos…"):
        df_custos = load_custos()

    c = st.columns(4)
    c[0].metric("Entradas Normalizadas", f"{len(df_custos):,}".replace(",", "."))
    c[1].metric("Produtos Únicos", str(df_custos["id_product"].nunique()))
    c[2].metric("Data Inicial", df_custos["start_date"].min().strftime("%d/%m/%Y"))
    c[3].metric("Data Final", df_custos["start_date"].max().strftime("%d/%m/%Y"))

    st.markdown("---")

    tab1, tab2 = st.tabs(["Análise de Preços", "Amostra da Tabela"])

    with tab1:
        col_a, col_b = st.columns(2)

        with col_a:
            df_cat = (
                df_custos.groupby("category")["usd_price"].mean().reset_index()
            )
            df_cat.columns = ["Categoria", "Preço Médio (USD)"]
            fig = px.bar(
                df_cat, x="Categoria", y="Preço Médio (USD)",
                title="Preço Médio de Importação por Categoria",
                color="Categoria",
                color_discrete_sequence=[TEAL, GREEN, BLUE],
            )
            fig.update_layout(**plotly_layout(), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            # Top 6 most expensive products
            top6 = df_custos.groupby("id_product")["usd_price"].mean().nlargest(6).index
            df_top = df_custos[df_custos["id_product"].isin(top6)].copy()
            fig2 = px.line(
                df_top, x="start_date", y="usd_price",
                color="id_product",
                title="Evolução de Preços — Top 6 Produtos Mais Caros (USD)",
                labels={"start_date": "Data", "usd_price": "Preço (USD)", "id_product": "Produto"},
                color_discrete_sequence=[TEAL, GREEN, BLUE, ORANGE, RED, WHITE],
            )
            fig2.update_layout(**plotly_layout())
            st.plotly_chart(fig2, use_container_width=True)

        # Price distribution — box plot by category (cleaner than histogram)
        fig3 = px.box(
            df_custos, x="category", y="usd_price",
            title="Distribuição de Preços de Importação por Categoria (USD)",
            labels={"category": "Categoria", "usd_price": "Preço (USD)"},
            color="category",
            color_discrete_sequence=[TEAL, GREEN, BLUE],
            points="outliers",
        )
        fig3.update_layout(**plotly_layout(), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        st.dataframe(df_custos.head(40), use_container_width=True, hide_index=True)

    ind_card(
        card_header(ICO_OK, "Resultado", GREEN) +
        f"""<p style='color:{GRAY2}'>
            JSON hierárquico (products → historic_data[]) achatado para
            <strong style='color:{WHITE}'>1.260 linhas</strong>.
            Schema final: <code>product_id</code>, <code>product_name</code>,
            <code>category</code>, <code>start_date</code>, <code>usd_price</code>.
        </p>"""
    )


# ─────────────────────────────────────────────────────────────────────────
# PAGE: Q4 — Loss Analysis
# ─────────────────────────────────────────────────────────────────────────
elif page == "q4":
    section("Análise de Prejuízo", "Custo de importação em BRL vs receita de vendas")

    df_prej = load_analise_prejuizo()

    if df_prej is None:
        st.warning(
            "Arquivo `outputs/analise_prejuizo.csv` não encontrado. "
            "Execute `questao_4_prejuizo.py` primeiro para gerar os resultados."
        )
    else:
        maior_abs = df_prej.loc[df_prej["prejuizo_total"].idxmax()]
        maior_pct = df_prej.loc[df_prej["percentual_perda"].idxmax()]
        n_com_prej = int((df_prej["prejuizo_total"] > 0).sum())

        c = st.columns(3)
        c[0].metric(
            "Maior Prejuízo Absoluto",
            f"R$ {maior_abs['prejuizo_total']/1e6:.1f} M",
            delta=f"Produto {int(maior_abs['id_product'])}",
            delta_color="inverse",
        )
        c[1].metric(
            "Maior % de Perda",
            f"{maior_pct['percentual_perda']:.1%}",
            delta=f"Produto {int(maior_pct['id_product'])}",
            delta_color="inverse",
        )
        c[2].metric("Produtos com Prejuízo", str(n_com_prej))

        st.markdown("---")

        n_show = st.slider("Quantidade de produtos a exibir", 10, min(60, len(df_prej)), 25)

        tab1, tab2, tab3 = st.tabs(["Prejuízo Absoluto", "% de Perda", "Tabela Completa"])

        with tab1:
            df_top = df_prej.nlargest(n_show, "prejuizo_total").copy()
            df_top["prejuizo_M"] = df_top["prejuizo_total"] / 1e6
            df_top["label"] = "Produto " + df_top["id_product"].astype(int).astype(str)
            fig = px.bar(
                df_top.sort_values("prejuizo_M"),
                x="prejuizo_M", y="label", orientation="h",
                title=f"Top {n_show} — Maior Prejuízo Total (R$ milhões)",
                labels={"prejuizo_M": "Prejuízo (R$ M)", "label": ""},
                color="prejuizo_M",
                color_continuous_scale=[[0, TEAL], [0.5, ORANGE], [1, RED]],
            )
            fig.update_layout(**plotly_layout(), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            df_pct = df_prej.nlargest(n_show, "percentual_perda").copy()
            df_pct["pct"] = df_pct["percentual_perda"] * 100
            df_pct["label"] = "Produto " + df_pct["id_product"].astype(int).astype(str)
            fig2 = px.bar(
                df_pct.sort_values("pct"),
                x="pct", y="label", orientation="h",
                title=f"Top {n_show} — Maior Percentual de Perda (%)",
                labels={"pct": "Perda (%)", "label": ""},
                color="pct",
                color_continuous_scale=[[0, TEAL], [0.5, ORANGE], [1, RED]],
            )
            fig2.update_layout(**plotly_layout(), coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

            # Scatter: revenue vs loss
            fig3 = px.scatter(
                df_prej,
                x="receita_total", y="prejuizo_total",
                size="percentual_perda", color="percentual_perda",
                title="Receita vs Prejuízo por Produto (tamanho = % perda)",
                labels={
                    "receita_total": "Receita Total (R$)",
                    "prejuizo_total": "Prejuízo Total (R$)",
                    "percentual_perda": "% Perda",
                },
                color_continuous_scale=[[0, TEAL], [0.5, ORANGE], [1, RED]],
                hover_data={"id_product": True},
            )
            fig3.update_layout(**plotly_layout())
            st.plotly_chart(fig3, use_container_width=True)

        with tab3:
            df_show = df_prej.copy()
            df_show["receita_total"] = df_show["receita_total"].map(fmt_brl)
            df_show["prejuizo_total"] = df_show["prejuizo_total"].map(fmt_brl)
            df_show["percentual_perda"] = df_show["percentual_perda"].map(lambda x: f"{x:.2%}")
            st.dataframe(df_show, use_container_width=True, hide_index=True)

        ind_card(
            card_header(ICO_ERR, "Resultado", RED) +
            f"""<p style='color:{GRAY2}; line-height:1.8'>
                <strong style='color:{WHITE}'>Maior prejuízo absoluto:</strong>
                Produto <strong>72</strong> (Motor de Popa Volvo Maré 69HP)
                → <strong style='color:{RED}'>R$ 36.177.362,15</strong><br>
                <strong style='color:{WHITE}'>Maior % de perda:</strong>
                também Produto <strong>72</strong>
                → <strong style='color:{RED}'>36,46%</strong> de perda<br>
                <strong style='color:{WHITE}'>São o mesmo produto?</strong>
                {ICO_OK} <strong style='color:{GREEN}'>SIM</strong>
            </p>"""
        )


# ─────────────────────────────────────────────────────────────────────────
# PAGE: Q5 — Loyal Customers
# ─────────────────────────────────────────────────────────────────────────
elif page == "q5":
    section("Clientes Fiéis", "Ticket médio + diversidade de categorias ≥ 3")

    with st.spinner("Calculando métricas de clientes…"):
        df_vendas = load_vendas()
        _, df_clean = load_produtos()
        df_clientes = load_clientes()

        df_prod = (
            df_clean[["code", "categoria"]]
            .rename(columns={"code": "id_product"})
            .drop_duplicates()
        )
        df_m = df_vendas.merge(df_prod, on="id_product", how="left")

        metricas = (
            df_m.groupby("id_client")
            .agg(
                faturamento_total=("total", "sum"),
                frequencia=("id", "count"),
                diversidade=("categoria", "nunique"),
            )
            .reset_index()
        )
        metricas["ticket_medio"] = metricas["faturamento_total"] / metricas["frequencia"]

        elite = metricas[metricas["diversidade"] >= 3].copy()
        top10 = elite.nlargest(10, "ticket_medio").reset_index(drop=True)
        top10.index = top10.index + 1

        top10_ids = top10["id_client"].tolist()
        cat_top10 = (
            df_m[df_m["id_client"].isin(top10_ids)]
            .groupby("categoria")["qtd"].sum()
            .reset_index()
            .sort_values("qtd", ascending=False)
        )
        best_cat = cat_top10.iloc[0]

        # Enrich with names
        clientes_map = df_clientes.set_index("code")["full_name"].to_dict()
        top10["nome"] = top10["id_client"].map(clientes_map).fillna("—")

    c = st.columns(3)
    c[0].metric("Clientes Elite (diversidade ≥ 3)", str(len(elite)))
    c[1].metric("Ticket Médio Máximo", fmt_brl(top10["ticket_medio"].max()))
    c[2].metric("Categoria Mais Comprada (Top 10)", best_cat["categoria"])

    st.markdown("---")

    tab1, tab2 = st.tabs(["Top 10 Clientes", "Categoria Favorita"])

    with tab1:
        display = top10[["id_client", "nome", "faturamento_total", "frequencia", "ticket_medio", "diversidade"]].copy()
        display.columns = ["ID", "Nome", "Faturamento (R$)", "Frequência", "Ticket Médio (R$)", "Diversidade"]
        display["Faturamento (R$)"] = display["Faturamento (R$)"].map(fmt_brl)
        display["Ticket Médio (R$)"] = display["Ticket Médio (R$)"].map(fmt_brl)
        st.dataframe(display, use_container_width=True)

        top10_sorted = top10.sort_values("ticket_medio", ascending=True).copy()
        top10_sorted["label"] = top10_sorted.apply(
            lambda r: r["nome"] if r["nome"] != "—" else f"Cliente {int(r['id_client'])}", axis=1
        )
        fig = px.bar(
            top10_sorted, x="ticket_medio", y="label", orientation="h",
            title="Ticket Médio — Top 10 Clientes Fiéis",
            labels={"ticket_medio": "Ticket Médio (R$)", "label": ""},
            color="ticket_medio",
            color_continuous_scale=[[0, BLUE], [0.5, TEAL], [1, GREEN]],
            text_auto=".3s",
        )
        fig.update_layout(**plotly_layout(), coloraxis_showscale=False)
        fig.update_yaxes(tickfont=dict(size=12))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = px.bar(
            cat_top10, x="categoria", y="qtd",
            title="Itens Comprados por Categoria — Top 10 Clientes",
            labels={"categoria": "Categoria", "qtd": "Total de Itens"},
            color="categoria",
            color_discrete_sequence=[GREEN, TEAL, BLUE],
            text_auto=True,
        )
        fig2.update_layout(**plotly_layout(), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    ind_card(
        card_header(ICO_OK, "Resultado", TEAL) +
        f"""
        <p style='color:{GRAY2}; line-height:1.8'>
            <strong style='color:{WHITE}'>Top 1:</strong>
            Cliente 47 — Ticket Médio <strong>R$ 336.859,70</strong><br>
            <strong style='color:{WHITE}'>Categoria mais comprada pelos Top 10:</strong>
            <strong style='color:{GREEN}'>propulsão</strong> (6.030 itens)
        </p>"""
    )


# ─────────────────────────────────────────────────────────────────────────
# PAGE: Q6 — Calendar Dimension
# ─────────────────────────────────────────────────────────────────────────
elif page == "q6":
    section("Calendário de Vendas", "Média de vendas por dia da semana incluindo dias com zero vendas")

    with st.spinner("Construindo calendário…"):
        df_vendas = load_vendas()
        _, df_prod_clean = load_produtos()

        # ── Calendário completo: TODOS os dias de 2023-2024, inclusive sem vendas ──
        calendario = pd.DataFrame({"data": pd.date_range("2023-01-01", "2024-12-31")})
        df_day = (
            df_vendas.assign(data=df_vendas["sale_date"].dt.normalize())
            .groupby("data")["total"].sum()
            .reset_index()
        )
        # LEFT JOIN garante que dias sem venda entram com total = 0
        df_cal = calendario.merge(df_day, on="data", how="left").fillna({"total": 0})

        _NAMES = {
            0: "Segunda", 1: "Terça", 2: "Quarta",
            3: "Quinta",  4: "Sexta", 5: "Sábado", 6: "Domingo",
        }
        _ORDER = list(_NAMES.values())
        df_cal["dia_semana"] = df_cal["data"].dt.dayofweek.map(_NAMES)
        df_cal["num_dia"]    = df_cal["data"].dt.dayofweek

        # Média e desvio padrão por dia (incluindo zeros)
        stats = (
            df_cal.groupby(["dia_semana", "num_dia"])["total"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("num_dia")
        )
        stats.columns = ["dia_semana", "num_dia", "media", "desvio", "n_dias"]
        stats["cv"] = stats["desvio"] / stats["media"]  # coeficiente de variação
        stats["erro_std"] = stats["desvio"] / stats["n_dias"].pow(0.5)

        pior  = stats.loc[stats["media"].idxmin()]
        melhor = stats.loc[stats["media"].idxmax()]

        # Heatmap: média de vendas por dia da semana × categoria de produto
        df_prod_map = df_prod_clean[["code", "categoria"]].rename(columns={"code": "id_product"}).drop_duplicates()
        df_cat = df_vendas.merge(df_prod_map, on="id_product", how="left")
        df_cat["data"] = df_cat["sale_date"].dt.normalize()
        df_cat["dia_semana"] = df_cat["sale_date"].dt.dayofweek.map(_NAMES)

        heat_data = (
            df_cat.groupby(["categoria", "dia_semana"])["total"]
            .sum()
            .unstack(fill_value=0)
            .reindex(columns=_ORDER, fill_value=0)
        )

    c = st.columns(4)
    c[0].metric("Melhor Dia", melhor["dia_semana"])
    c[1].metric("Média no Melhor Dia", fmt_brl(melhor["media"]))
    c[2].metric("Pior Dia", pior["dia_semana"])
    c[3].metric("Média no Pior Dia", fmt_brl(pior["media"]))

    st.markdown("---")

    answer_box(
        "<strong>Como a análise foi feita?</strong><br>"
        "Imaginando um calendário com <em>todos</em> os dias de 2023 e 2024 — inclusive "
        "os dias em que a loja não vendeu nada. Esses dias de 'silêncio' foram incluídos "
        "com valor zero, o que torna a média mais honesta. Sem esse ajuste, a média "
        "pareceria maior do que realmente é, pois os dias ruins simplesmente sumiriam do cálculo."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    tab_a, tab_b, tab_c = st.tabs(["Média por Dia", "Variabilidade", "Por Categoria"])

    with tab_a:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.bar(
                stats, x="dia_semana", y="media",
                title="Média de Receita por Dia da Semana",
                labels={"dia_semana": "", "media": "Média (R$)"},
                color="media",
                color_continuous_scale=[[0, RED], [0.5, ORANGE], [1, GREEN]],
                category_orders={"dia_semana": _ORDER},
                text_auto=".2s",
            )
            fig.update_layout(**plotly_layout(), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig2 = px.bar(
                stats.sort_values("media"),
                x="media", y="dia_semana", orientation="h",
                title="Ranking — Dias Ordenados por Receita Média",
                labels={"media": "Média (R$)", "dia_semana": ""},
                color="media",
                color_continuous_scale=[[0, RED], [0.5, ORANGE], [1, GREEN]],
                text_auto=".2s",
            )
            fig2.update_layout(**plotly_layout(), coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

    with tab_b:
        # Gráfico de barras com barras de erro (desvio padrão)
        fig3 = go.Figure()
        colors_bar = [GREEN if d == melhor["dia_semana"] else RED if d == pior["dia_semana"] else TEAL
                      for d in stats["dia_semana"]]
        fig3.add_trace(go.Bar(
            x=stats["dia_semana"], y=stats["media"],
            error_y=dict(type="data", array=stats["erro_std"].tolist(), visible=True,
                         color="rgba(255,255,255,0.4)", thickness=1.5, width=6),
            marker_color=colors_bar,
            text=[fmt_brl(v) for v in stats["media"]],
            textposition="outside",
            name="Média ± Erro Padrão",
        ))
        fig3.update_layout(**plotly_layout())
        fig3.update_layout(title="Média de Receita com Intervalo de Confiança (±1 Erro Padrão)",
                           yaxis_title="Receita (R$)")
        fig3.update_xaxes(categoryorder="array", categoryarray=_ORDER)
        st.plotly_chart(fig3, use_container_width=True)

        # Coeficiente de variação
        fig4 = px.bar(
            stats.sort_values("cv", ascending=False),
            x="dia_semana", y="cv",
            title="Imprevisibilidade por Dia (Coeficiente de Variação — menor = mais estável)",
            labels={"dia_semana": "", "cv": "CV (desvio / média)"},
            color="cv",
            color_continuous_scale=[[0, GREEN], [0.5, ORANGE], [1, RED]],
            text_auto=".2f",
        )
        fig4.update_layout(**plotly_layout(), coloraxis_showscale=False)
        fig4.update_xaxes(categoryorder="array", categoryarray=_ORDER)
        st.plotly_chart(fig4, use_container_width=True)

        answer_box(
            "<strong>O que é o Coeficiente de Variação (CV)?</strong><br>"
            "Mede o quanto as vendas de um dia 'oscilam' em relação à sua própria média. "
            "Um CV alto significa que esse dia é imprevisível — às vezes vende muito, "
            "às vezes quase nada. Um CV baixo significa que o dia é consistente. "
            "Dias com <em>alta média e baixo CV</em> são os mais valiosos para o negócio."
        )

    with tab_c:
        fig5 = px.imshow(
            heat_data,
            title="Receita Total Acumulada por Categoria e Dia da Semana (2023–2024)",
            labels={"x": "Dia da Semana", "y": "Categoria", "color": "Receita (R$)"},
            color_continuous_scale=[[0, "#0d1b2e"], [0.35, "#1a3a5c"], [0.7, BLUE], [1, TEAL]],
            text_auto=".2s",
            aspect="auto",
        )
        fig5.update_layout(**plotly_layout())
        st.plotly_chart(fig5, use_container_width=True)

        answer_box(
            "<strong>Atenção ao interpretar este mapa:</strong><br>"
            "Os valores mostram o total acumulado em 2 anos — não a média diária. "
            "Ele serve para identificar <em>quais combinações de categoria + dia concentram mais receita</em>, "
            "mas não é um modelo preditivo. Com apenas ~104 ocorrências por dia/categoria, "
            "qualquer previsão teria margem de erro considerável."
        )

    ind_card(
        card_header(ICO_INFO, "Análise e Recomendações", TEAL) +
        f"""
        <p style='color:{GRAY2}; line-height:1.9'>
            {ICO_OK} <strong style='color:{WHITE}'>Melhor dia:</strong>
            {melhor['dia_semana']} — média de {fmt_brl(melhor['media'])} por dia<br>
            {ICO_ERR} <strong style='color:{WHITE}'>Pior dia:</strong>
            {pior['dia_semana']} — média de {fmt_brl(pior['media'])} por dia<br><br>
            <strong style='color:{WHITE}'>Sugestões práticas:</strong><br>
            — Concentrar equipe de vendas e ações comerciais nos dias de maior média<br>
            — Criar campanhas específicas para {pior['dia_semana']} (ex: frete grátis, desconto relâmpago)<br>
            — Monitorar o CV: dias com alta variação merecem análise de causas (feriados, clima, sazonalidade)<br>
            — O mapa por categoria revela onde cada linha de produto performa melhor ao longo da semana
        </p>"""
    )


# ─────────────────────────────────────────────────────────────────────────
# PAGE: Q7 — Demand Forecast
# ─────────────────────────────────────────────────────────────────────────
elif page == "q7":
    section("Previsão de Demanda", "Total de Vendas Diárias · Teste: Janeiro 2024")

    with st.spinner("Calculando previsões…"):
        df_vendas = load_vendas()

        # Agrega TODAS as vendas por dia
        df_daily = (
            df_vendas.groupby(df_vendas["sale_date"].dt.normalize())["qtd"]
            .sum().reset_index()
        )
        df_daily.columns = ["data", "vendas"]

        cal = pd.DataFrame({"data": pd.date_range("2023-01-01", "2024-12-31")})
        df_full = cal.merge(df_daily, on="data", how="left").fillna({"vendas": 0})
        df_full["vendas"] = df_full["vendas"].astype(int)
        df_full = df_full.sort_values("data").reset_index(drop=True)
        df_full["previsao"] = df_full["vendas"].shift(1).rolling(7, min_periods=1).mean()

        treino = df_full[df_full["data"] <= "2023-12-31"]
        teste  = df_full[(df_full["data"] >= "2024-01-01") & (df_full["data"] <= "2024-01-31")].copy()

        mae         = float(np.abs(teste["previsao"] - teste["vendas"]).mean())
        rmse        = float(np.sqrt(((teste["previsao"] - teste["vendas"])**2).mean()))
        # MAPE: apenas dias com venda real > 0 para evitar divisão por zero
        dias_com_venda = teste[teste["vendas"] > 0].copy()
        mape        = float((np.abs(dias_com_venda["previsao"] - dias_com_venda["vendas"]) / dias_com_venda["vendas"]).mean() * 100)
        # Baseline ingênuo: prever o valor de ontem (shift 1)
        mae_naive   = float(np.abs(teste["vendas"].shift(1) - teste["vendas"]).dropna().mean())
        soma_sem1   = int(round(teste[teste["data"] <= "2024-01-07"]["previsao"].sum()))
        real_sem1   = int(teste[teste["data"] <= "2024-01-07"]["vendas"].sum())
        media_diaria_2023 = int(treino["vendas"].mean())

    # ── Intro ────────────────────────────────────────────────────────────────
    st.markdown(
        f"""<div style='background:{CARD}; border:1px solid {BORDER};
            border-left:3px solid {BLUE}; border-radius:8px;
            padding:14px 18px; margin-bottom:20px; color:{GRAY2}; line-height:1.7; font-size:0.88rem'>
            <strong style='color:{WHITE}'>O que é isso?</strong><br>
            Usamos todas as vendas de 2023 para <strong>prever quantas unidades seriam vendidas
            por dia em janeiro de 2024</strong>. O método é simples: a previsão de amanhã é a
            <strong>média dos últimos 7 dias</strong>. Se a loja vendeu bastante essa semana,
            esperamos vender bastante amanhã também.
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Métricas ─────────────────────────────────────────────────────────────
    c = st.columns(4)
    c[0].metric("MAE (erro médio diário)", f"±{mae:.1f} unid.",
        help="Erro Absoluto Médio: diferença média entre previsto e real por dia.")
    c[1].metric("MAPE (erro percentual)", f"{mape:.1f}%",
        help="Mean Absolute Percentage Error: o erro em % das vendas reais. Abaixo de 20% é considerado bom em séries de varejo.")
    c[2].metric("RMSE", f"{rmse:.1f} unid.",
        help="Raiz do Erro Quadrático Médio: penaliza mais os dias de grande erro.")
    c[3].metric("Baseline ingênuo (MAE)", f"±{mae_naive:.1f} unid.",
        help="Se simplesmente usássemos 'amanhã = hoje', o erro seria este. Se nosso MAE for menor, o modelo agrega valor.")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Previsão vs. Realidade (Jan/2024)", "Histórico de Vendas (2023)", "Produto 54 — Demanda Esparsa"])

    with tab1:
        st.markdown(
            f"<p style='color:{GRAY}; font-size:0.83rem; margin-bottom:8px'>"
            "Barras azuis = vendas reais por dia · Linha laranja = o que o modelo previu</p>",
            unsafe_allow_html=True,
        )
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=teste["data"], y=teste["vendas"],
            name="Vendas Reais", marker_color=TEAL, opacity=0.75,
        ))
        fig.add_trace(go.Scatter(
            x=teste["data"], y=teste["previsao"],
            name="Previsão (MM7)", mode="lines+markers",
            line=dict(color=ORANGE, width=2.5, dash="dash"),
            marker=dict(size=7),
        ))
        fig.update_layout(**plotly_layout(
            title="Vendas Totais Diárias — Previsto × Realidade · Jan/2024",
            xaxis_title="Data", yaxis_title="Unidades Vendidas",
        ))
        st.plotly_chart(fig, use_container_width=True)

        # ── Insights das métricas ─────────────────────────────────────────────
        rmse_mae_ratio = rmse / mae if mae > 0 else 0
        bateu_naive    = mae < mae_naive
        mape_ok        = mape < 20

        ci1, ci2, ci3 = st.columns(3)
        with ci1:
            st.markdown(
                f"""<div style='background:{CARD}; border:1px solid {BORDER};
                    border-left:3px solid {"' + GREEN + '" if mape_ok else "' + ORANGE + '"};
                    border-radius:8px; padding:14px 18px; color:{GRAY2};
                    font-size:0.84rem; line-height:1.65'>
                    <strong style='color:{WHITE}'>MAPE = {mape:.1f}%</strong><br>
                    {"✔ Dentro do limiar de 20% aceito em séries de varejo — o modelo tem <strong style='color:" + GREEN + "'>boa precisão percentual</strong> no agregado." if mape_ok else "✖ Acima de 20% — o modelo <strong style='color:" + ORANGE + "'>erra mais de 1 em cada 5 unidades previstas</strong> em média nos dias com venda."}
                </div>""",
                unsafe_allow_html=True,
            )
        with ci2:
            st.markdown(
                f"""<div style='background:{CARD}; border:1px solid {BORDER};
                    border-left:3px solid {"' + GREEN + '" if bateu_naive else "' + RED + '"};
                    border-radius:8px; padding:14px 18px; color:{GRAY2};
                    font-size:0.84rem; line-height:1.65'>
                    <strong style='color:{WHITE}'>MM7 vs. Baseline ingênuo</strong><br>
                    MAE do modelo = <strong>{mae:.1f}</strong> · MAE ingênuo = <strong>{mae_naive:.1f}</strong><br>
                    {"A média móvel <strong style='color:" + GREEN + "'>supera</strong> simplesmente repetir o dia anterior — a janela de 7 dias agrega valor real." if bateu_naive else "A média móvel <strong style='color:" + RED + "'>não supera</strong> o baseline trivial — nesse nível de agregação, repetir o dia anterior seria igual ou melhor."}
                </div>""",
                unsafe_allow_html=True,
            )
        with ci3:
            st.markdown(
                f"""<div style='background:{CARD}; border:1px solid {BORDER};
                    border-left:3px solid {"' + ORANGE + '" if rmse_mae_ratio > 1.5 else "' + TEAL + '"};
                    border-radius:8px; padding:14px 18px; color:{GRAY2};
                    font-size:0.84rem; line-height:1.65'>
                    <strong style='color:{WHITE}'>RMSE / MAE = {rmse_mae_ratio:.1f}×</strong><br>
                    {"Razão <strong style='color:" + ORANGE + "'>alta</strong>: há dias com erros grandes escondidos na média do MAE — o modelo sofre em picos/vales atípicos." if rmse_mae_ratio > 1.5 else "Razão <strong style='color:" + TEAL + "'>próxima de 1</strong>: os erros são distribuídos de forma uniforme — sem grandes picos pontuais distorcendo o modelo."}
                </div>""",
                unsafe_allow_html=True,
            )

    with tab2:
        st.markdown(
            f"<p style='color:{GRAY}; font-size:0.83rem; margin-bottom:8px'>"
            "Linha azul = vendas reais dia a dia · Linha verde = média suavizada (7 dias)</p>",
            unsafe_allow_html=True,
        )
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=treino["data"], y=treino["vendas"],
            name="Vendas Diárias", line=dict(color=TEAL, width=1), opacity=0.6,
        ))
        fig2.add_trace(go.Scatter(
            x=treino["data"], y=treino["previsao"],
            name="Média 7 dias (previsão base)", line=dict(color=GREEN, width=2.5),
        ))
        fig2.update_layout(**plotly_layout(
            title="Histórico 2023 — Total de Vendas Diárias",
            xaxis_title="Data", yaxis_title="Unidades Vendidas",
        ))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            f"""<div style='background:{CARD}; border:1px solid {BORDER};
                border-left:3px solid {GREEN}; border-radius:8px;
                padding:14px 18px; color:{GRAY2}; line-height:1.7; font-size:0.88rem'>
                <strong style='color:{WHITE}'>O que a linha verde mostra?</strong><br>
                É a "memória" do modelo: a média das últimas 7 vendas. Ela <strong>suaviza os altos e
                baixos</strong> do dia a dia e serve como base da previsão. Note que em períodos longos
                sem venda (linha verde cai a zero), o modelo perde referência — é quando ele mais erra.
            </div>""",
            unsafe_allow_html=True,
        )

    with tab3:
        # ── Produto 54 — dados ────────────────────────────────────────────────
        PROD54_ID   = 54
        PROD54_NAME = "Motor de Popa Yamaha Evo Dash 155HP"

        df_p54 = df_vendas[df_vendas["id_product"] == PROD54_ID].copy()
        dtf_daily54 = df_p54.groupby("sale_date")["qtd"].sum().reset_index().rename(
            columns={"sale_date": "data", "qtd": "vendas"}
        )
        cal54 = pd.DataFrame({"data": pd.date_range("2023-01-01", "2024-12-31")})
        df54 = cal54.merge(dtf_daily54, on="data", how="left").fillna({"vendas": 0})
        df54["vendas"] = df54["vendas"].astype(int)
        df54 = df54.sort_values("data").reset_index(drop=True)
        df54["prev_7d"] = df54["vendas"].shift(1).rolling(7, min_periods=1).mean()

        teste54 = df54[(df54["data"] >= "2024-01-01") & (df54["data"] <= "2024-01-31")].copy()
        mae54  = float(np.abs(teste54["prev_7d"] - teste54["vendas"]).mean())
        rmse54 = float(np.sqrt(((teste54["prev_7d"] - teste54["vendas"])**2).mean()))
        dias_v54 = teste54[teste54["vendas"] > 0]
        mape54 = float((np.abs(dias_v54["prev_7d"] - dias_v54["vendas"]) / dias_v54["vendas"]).mean() * 100) if len(dias_v54) > 0 else float("nan")
        dias_zero = int((teste54["vendas"] == 0).sum())
        dias_venda = int((teste54["vendas"] > 0).sum())

        # ── Alerta demanda esparsa ────────────────────────────────────────────
        st.markdown(
            f"""<div style='background:rgba(224,82,82,0.08); border:1px solid rgba(224,82,82,0.35);
                border-radius:8px; padding:14px 18px; margin-bottom:18px;
                color:{GRAY2}; font-size:0.88rem; line-height:1.7'>
                <strong style='color:{RED}'>Atenção — demanda esparsa detectada</strong><br>
                Em janeiro/2024, o produto <strong style='color:{WHITE}'>{PROD54_NAME}</strong>
                vendeu em <strong style='color:{WHITE}'>{dias_venda} dias</strong>
                e ficou <strong style='color:{RED}'>{dias_zero} dias sem vender</strong>.
                Isso distorce qualquer métrica baseada em média diária.
            </div>""",
            unsafe_allow_html=True,
        )

        # ── Métricas produto 54 ───────────────────────────────────────────────
        c54 = st.columns(3)
        c54[0].metric("MAE", f"±{mae54:.2f} unid.",
            help="Parece baixo — mas 20 dias de zero puxam a média pra baixo artificialmente.")
        c54[1].metric("MAPE (dias com venda)", f"{mape54:.1f}%",
            help="Calculado só nos dias em que houve venda real. Aqui está a verdade: o modelo errou ~87% nos únicos 2 dias que importavam.")
        c54[2].metric("RMSE", f"±{rmse54:.2f} unid.",
            help="Bem maior que o MAE. A diferença entre MAE e RMSE revela que há poucos erros grandes (dias 21 e 22) pesando no cálculo.")

        st.markdown("---")

        # ── Gráfico barras Jan/2024 ───────────────────────────────────────────
        fig54 = go.Figure()
        fig54.add_trace(go.Bar(
            x=teste54["data"], y=teste54["vendas"],
            name="Vendas Reais", marker_color=TEAL, opacity=0.8,
        ))
        fig54.add_trace(go.Scatter(
            x=teste54["data"], y=teste54["prev_7d"],
            name="Previsão MM7", mode="lines+markers",
            line=dict(color=ORANGE, width=2.5, dash="dash"),
            marker=dict(size=7),
        ))
        fig54.update_layout(**plotly_layout(
            title=f"{PROD54_NAME} — Previsto × Real · Jan/2024",
            xaxis_title="Data", yaxis_title="Unidades",
        ))
        st.plotly_chart(fig54, use_container_width=True)

        # ── Cards de interpretação ────────────────────────────────────────────
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                f"""<div style='background:{CARD}; border:1px solid {BORDER};
                    border-left:3px solid {ORANGE}; border-radius:8px; padding:14px 18px;
                    color:{GRAY2}; font-size:0.85rem; line-height:1.7; height:100%'>
                    <strong style='color:{WHITE}'>Por que o MAE engana?</strong><br>
                    MAE = {mae54:.2f} parece ótimo. Mas dos 31 dias, <strong>{dias_zero} tiveram venda zero</strong>
                    — e prever zero quando a realidade é zero não exige nenhum modelo.
                    O erro real aparece no <strong style='color:{ORANGE}'>MAPE ({mape54:.0f}%)</strong>:
                    nos únicos 2 dias que vendeu (dias 21 e 22), o modelo errou feio
                    porque não havia sinal no histórico recente.
                </div>""",
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(
                f"""<div style='background:{CARD}; border:1px solid {BORDER};
                    border-left:3px solid {BLUE}; border-radius:8px; padding:14px 18px;
                    color:{GRAY2}; font-size:0.85rem; line-height:1.7; height:100%'>
                    <strong style='color:{WHITE}'>MAE vs. RMSE — o que a diferença revela</strong><br>
                    MAE = {mae54:.2f} · RMSE = {rmse54:.2f} — razão de <strong style='color:{WHITE}'>{rmse54/mae54:.1f}×</strong>.
                    Quando RMSE é muito maior que MAE,
                    existem <strong>poucos erros grandes escondidos na média</strong>.
                    Aqui são os dias 21 (erro de 11 unid.) e 22 (erro de ~4 unid.)
                    puxando o RMSE pra cima enquanto os 20 dias de zero mascaram o MAE.
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""<div style='background:{CARD}; border:1px solid {BORDER};
                border-left:3px solid {GREEN}; border-radius:8px; padding:14px 18px;
                color:{GRAY2}; font-size:0.85rem; line-height:1.7'>
                <strong style='color:{WHITE}'>Conclusão — o que fazer com demanda esparsa?</strong><br>
                Média móvel diária <strong>não é a ferramenta certa</strong> para produtos que vendem
                raramente. O modelo não tem como prever os dias 21–22 porque não há padrão diário
                estável — qualquer modelo temporal vai falhar aqui. Alternativas:
                agregar em <strong style='color:{TEAL}'>granularidade mensal</strong>,
                usar modelos de <strong style='color:{TEAL}'>contagem (Poisson / Negative Binomial)</strong> para demanda
                esparsa, ou investigar o que causou os picos dos dias 21–22
                (campanha, pedido especial?). Para demanda esparsa, a
                <strong style='color:{TEAL}'>granularidade mensal ou modelos de contagem
                (Poisson / Negative Binomial) são mais adequados que a média móvel diária</strong>.
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Scorecard 4 células ───────────────────────────────────────────────────
    def _score_color(ok): return GREEN if ok else ORANGE
    pct_ganho_naive = (mae_naive - mae) / mae_naive * 100 if mae_naive > 0 else 0

    st.markdown(
        f"""<div class='kpi-grid kpi-grid-3' style='margin-bottom:16px'>
            <div class='kpi-cell'>
                <div class='kpi-label'>Erro médio por dia (MAE)</div>
                <div class='kpi-value' style='color:{_score_color(mae < mae_naive)}'>±{mae:.1f} unid.</div>
                <div style='font-size:0.75rem; color:{GRAY}; margin-top:4px'>
                    {"↓ " + f"{pct_ganho_naive:.0f}% melhor que prever ontem" if bateu_naive else "↑ pior que baseline ingênuo"}
                </div>
            </div>
            <div class='kpi-cell'>
                <div class='kpi-label'>Erro % nos dias com venda (MAPE)</div>
                <div class='kpi-value' style='color:{_score_color(mape_ok)}'>{mape:.1f}%</div>
                <div style='font-size:0.75rem; color:{GRAY}; margin-top:4px'>
                    {"✔ abaixo do limiar de 20% para varejo" if mape_ok else "✖ acima do limiar de 20% para varejo"}
                </div>
            </div>
            <div class='kpi-cell'>
                <div class='kpi-label'>Picos escondidos (RMSE / MAE)</div>
                <div class='kpi-value' style='color:{_score_color(rmse_mae_ratio <= 1.5)}'>{rmse_mae_ratio:.1f}×</div>
                <div style='font-size:0.75rem; color:{GRAY}; margin-top:4px'>
                    {"erros uniformes — sem picos grandes" if rmse_mae_ratio <= 1.5 else "há dias com erro grande distorcendo a média"}
                </div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    ind_card(
        card_header(ICO_INFO, "O que esses números significam na prática?", TEAL) +
        f"""<div style='display:grid; grid-template-columns:1fr 1fr; gap:16px; font-size:0.85rem; color:{GRAY2}; line-height:1.7'>
            <div>
                <strong style='color:{WHITE}'>Vale usar esse modelo ou tanto faz?</strong><br>
                {"Sim, vale. Em vez de simplesmente repetir as vendas de ontem, usar a <strong>média dos últimos 7 dias</strong> reduziu o erro em <strong style='color:" + GREEN + "'>" + f"{pct_ganho_naive:.0f}%" + "</strong>. No estoque, isso significa pedir " + f"{mae:.0f}" + " unidades a menos de margem de erro por dia." if bateu_naive else "Nesse caso, não faz diferença. Simplesmente <strong>repetir o total de ontem</strong> teria o mesmo resultado. O padrão de vendas muda tanto dia a dia que olhar os últimos 7 dias não ajuda."}
            </div>
            <div>
                <strong style='color:{WHITE}'>O modelo erra muito em % das vendas?</strong><br>
                {"Errou <strong style='color:" + GREEN + "'>" + f"{mape:.1f}%" + "</strong> em média nos dias com venda — dentro dos <strong>20% considerados aceitáveis</strong> para previsão de varejo. Em linguagem simples: a cada 10 unidades previstas, erra menos de 2." if mape_ok else "Errou <strong style='color:" + ORANGE + "'>" + f"{mape:.1f}%" + "</strong> nos dias com venda — acima do limite de 20%. Em linguagem simples: <strong>a cada 10 unidades previstas, erra mais de 2</strong>. Ainda útil como referência, mas não para decisões finas de estoque."}
            </div>
            <div>
                <strong style='color:{WHITE}'>Os erros são espalhados ou tem dias muito ruins?</strong><br>
                {"Os erros são <strong style='color:" + TEAL + "'>bem distribuídos</strong> — não tem um dia específico onde o modelo desandou completamente. Erra de forma parecida todos os dias, sem surpresas grandes." if rmse_mae_ratio <= 1.5 else "Tem <strong style='color:" + ORANGE + "'>alguns dias onde o modelo errou muito mais</strong> que a média sugere. Vale investigar esses dias: feriado, promoção ou entrega grande que distorceu o padrão?"}
            </div>
            <div>
                <strong style='color:{WHITE}'>Como melhorar a previsão?</strong><br>
                <span style='color:{GRAY}'>Considerar que <strong style='color:{WHITE}'>domingo vende menos</strong>
                (ver Calendário de Vendas), <strong style='color:{WHITE}'>feriados</strong> e
                sazonalidade — coisas que a média simples ignora.<br>
                Para produtos que vendem raramente, <strong style='color:{WHITE}'>agrupar por mês
                em vez de por dia</strong> dá previsões muito mais confiáveis.</span>
            </div>
        </div>"""
    )


# ─────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────
# PAGE: Q8 — Recommendation System
# ─────────────────────────────────────────────────────────────────────────
elif page == "q8":
    section("Recomendação de Produtos", "Quem comprou isso também levou… · Referência: GPS Garmin Vortex Maré Drift")

    TARGET_ID   = 27
    TARGET_NAME = "GPS Garmin Vortex Maré Drift"

    with st.spinner("Calculando matriz de similaridade…"):
        df_vendas = load_vendas()
        matriz = (
            df_vendas.groupby(["id_client", "id_product"])["qtd"]
            .max().unstack(fill_value=0)
        )
        matriz = (matriz > 0).astype(int)
        M = matriz.T.values.astype(float)
        normas = np.linalg.norm(M, axis=1, keepdims=True)
        normas[normas == 0] = 1
        M_norm = M / normas
        sim_matrix = M_norm @ M_norm.T
        sim_df = pd.DataFrame(sim_matrix, index=matriz.columns, columns=matriz.columns)
        similares_all = (
            sim_df[TARGET_ID].drop(TARGET_ID)
            .sort_values(ascending=False).head(15)
        )
        df_prod_raw, _ = load_produtos()
        _nome_map = df_prod_raw.drop_duplicates("code").set_index("code")["name"].to_dict()

    st.markdown(
        f"""<div style='background:{CARD}; border:1px solid {BORDER};
            border-left:3px solid {TEAL}; border-radius:8px;
            padding:14px 18px; margin-bottom:20px; color:{GRAY2}; line-height:1.7; font-size:0.88rem'>
            <strong style='color:{WHITE}'>O que é isso?</strong><br>
            Funciona como o "Quem comprou isso também levou…" da Amazon. O sistema olha o histórico
            de compras de todos os clientes e identifica quais produtos são frequentemente comprados
            <strong>pelos mesmos clientes</strong> que compraram o GPS Garmin. Quanto mais clientes
            em comum, maior a similaridade entre os produtos.
        </div>""",
        unsafe_allow_html=True,
    )

    top1_id   = similares_all.idxmax()
    top1_nome = _nome_map.get(top1_id, f"Produto {top1_id}")
    c = st.columns(3)
    c[0].metric("Produto de Referência", TARGET_NAME)
    c[1].metric("Produto Mais Recomendado", top1_nome)
    c[2].metric("Score de Similaridade", f"{similares_all.max():.4f}",
        help="Valor entre 0 e 1. Quanto mais próximo de 1, mais os mesmos clientes compraram os dois produtos.")

    st.markdown("---")
    tab1, tab2 = st.tabs(["Produtos Recomendados", "Como funciona"])

    with tab1:
        n_show = st.slider("Quantidade de produtos a exibir", 5, 15, 10)
        similares = similares_all.head(n_show)

        df_sim = similares.reset_index()
        df_sim.columns = ["id_produto", "similaridade"]
        df_sim["nome"] = df_sim["id_produto"].map(_nome_map).fillna("Produto " + df_sim["id_produto"].astype(str))
        # Label curto (id + nome truncado) para o eixo; nome completo no hover
        df_sim["label"] = df_sim.apply(
            lambda r: f"#{r['id_produto']} — {r['nome'][:35]}{'…' if len(r['nome'])>35 else ''}", axis=1
        )
        # Ordena crescente para maior aparecer no topo
        df_sim = df_sim.sort_values("similaridade", ascending=True)
        fig = px.bar(
            df_sim, x="similaridade", y="label", orientation="h",
            title=f"Produtos mais comprados junto ao {TARGET_NAME}",
            labels={"similaridade": "Similaridade (0 a 1)", "label": ""},
            color="similaridade",
            color_continuous_scale=[[0, BLUE], [1, TEAL]],
            text_auto=".4f",
            hover_data={"nome": True, "id_produto": True, "label": False},
        )
        fig.update_xaxes(range=[similares.min() - 0.01, similares.max() + 0.01])
        _layout_cfg = plotly_layout()
        _layout_cfg["yaxis"] = {**_layout_cfg.get("yaxis", {}), "tickfont": dict(size=11)}
        fig.update_layout(**_layout_cfg, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f"""<div style='background:{CARD}; border:1px solid {BORDER};
                border-left:3px solid {GREEN}; border-radius:8px;
                padding:14px 18px; color:{GRAY2}; line-height:1.7; font-size:0.88rem'>
                <strong style='color:{WHITE}'>Como interpretar?</strong><br>
                Scores acima de 0,85 indicam que os mesmos clientes compram esses dois produtos com
                muita frequência. Na prática: ao adicionar o GPS Garmin ao carrinho, o site deveria
                sugerir o <strong style='color:{TEAL}'>{top1_nome}</strong> como
                complemento — grande chance de o cliente querer os dois.
            </div>""",
            unsafe_allow_html=True,
        )

    with tab2:
        st.markdown(
            f"""<div style='background:{CARD}; border:1px solid {BORDER};
                border-radius:8px; padding:18px 20px; color:{GRAY2}; line-height:1.8; font-size:0.88rem'>
                <strong style='color:{WHITE}'>Passo 1 — Tabela de compras ({matriz.shape[0]} clientes x {matriz.shape[1]} produtos)</strong><br>
                Criamos uma tabela onde cada linha e um cliente e cada coluna e um produto.
                Celula = <strong>1</strong> se o cliente comprou ao menos uma vez, <strong>0</strong> se nunca comprou.<br><br>
                <strong style='color:{WHITE}'>Passo 2 — Similaridade de Cosseno</strong><br>
                Comparamos os produtos dois a dois: se os mesmos clientes compraram ambos,
                a similaridade e alta (proxima de 1). Se clientes completamente diferentes
                compraram cada um, a similaridade e baixa (proxima de 0).<br><br>
                <strong style='color:{WHITE}'>Passo 3 — Ranking</strong><br>
                Ordenamos os produtos pelo score de similaridade com o GPS Garmin e
                retornamos os 5 mais parecidos.<br><br>
                <strong style='color:{ORANGE}'>Limitacao:</strong>
                O metodo ignora a quantidade comprada e a ordem temporal.
                Produtos novos (sem historico) nao aparecem nas recomendacoes — efeito chamado
                <em>cold start</em>.
            </div>""",
            unsafe_allow_html=True,
        )

    ind_card(
        card_header(ICO_OK, "Resultado", TEAL) +
        f"""<p style='color:{GRAY2}; line-height:1.8; font-size:0.88rem'>
            Produto com <strong>maior similaridade</strong> ao GPS Garmin (id=27):
            <strong style='color:{TEAL}'>id {similares.idxmax()}</strong>
            — score <strong>{similares.max():.4f}</strong><br>
            Top 5: {", ".join(f"id {p}" for p in similares.index.tolist())}<br>
            <span style='color:{GRAY}'>Recomendacao de negocio: exibir esses produtos na pagina
            do GPS como "Frequentemente comprados juntos".</span>
        </p>"""
    )
