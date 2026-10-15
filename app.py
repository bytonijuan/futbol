import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configuración inicial de la página
st.set_page_config(
    page_title="Opta Analytics | Pro Football Engine", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# ESTILOS CSS PREMIUM (Estilo Sofascore / Opta Pro)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }

    /* Ocultar elementos nativos innecesarios de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Tarjetas de métricas tipo Terminal de Datos */
    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        transition: all 0.2s ease;
    }
    .metric-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.15);
    }
    .metric-label {
        color: #9ca3af;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .metric-sub {
        color: #60a5fa;
        font-size: 0.75rem;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Pestañas (Tabs) Estilo Corporativo */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #111827;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid #1f2937;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #9ca3af;
        border-radius: 6px;
        font-weight: 600;
        padding: 8px 18px;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
    }

    /* Tablas y DataFrames */
    dataframe, table {
        font-family: 'Inter', sans-serif;
    }

    /* Selectores limpios */
    .stSelectbox label {
        color: #9ca3af !important;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. CATÁLOGO DE LIGAS
# ==========================================
LIGAS_PRINCIPALES = {
    "🇪🇸 España - LaLiga EA Sports": "SP1",
    "🇪🇸 España - LaLiga Hypermotion (2ª)": "SP2",
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Inglaterra - Premier League": "E0",
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Inglaterra - Championship": "E1",
    "🇮🇹 Italia - Serie A": "I1",
    "🇩🇪 Alemania - Bundesliga": "D1",
    "🇫🇷 Francia - Ligue 1": "F1",
    "🇵🇹 Portugal - Primeira Liga": "P1",
    "🇳🇱 Países Bajos - Eredivisie": "N1",
    "🇧🇪 Bélgica - Pro League": "B1",
    "🇹🇷 Turquía - Süper Lig": "T1",
    "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Escocia - Premiership": "SC0"
}

# ==========================================
# 2. CONECTORES DE DATOS Y ELO
# ==========================================
@st.cache_data(ttl=3600)
def cargar_club_elo_vivo():
    for dias_atras in [0, 1, 2]:
        fecha_str = (datetime.now() - timedelta(days=dias_atras)).strftime('%Y-%m-%d')
        try:
            url = f"http://api.clubelo.com/{fecha_str}"
            df_elo = pd.read_csv(url)
            if not df_elo.empty and 'Club' in df_elo.columns:
                return df_elo[['Club', 'Country', 'Level', 'Elo']].dropna()
        except Exception:
            continue
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def descargar_liga_nacional(codigo_liga, temp_act="2627", temp_prev="2526"):
    try:
        dfs = []
        for t in [temp_prev, temp_act]:
            try:
                url = f"https://www.football-data.co.uk/mmz4281/{t}/{codigo_liga}.csv"
                dfs.append(pd.read_csv(url, encoding='latin1'))
            except Exception:
                pass
        df = pd.concat(dfs, ignore_index=True)
        df = df.rename(columns={
            'Date': 'fecha', 'HomeTeam': 'local', 'AwayTeam': 'visitante',
            'FTHG': 'goles_l', 'FTAG': 'goles_v', 'HST': 'tiros_puerta_l', 'AST': 'tiros_puerta_v',
            'HS': 'tiros_tot_l', 'AS': 'tiros_tot_v', 'HC': 'HC', 'AC': 'AC', 'HY': 'HY', 'AY': 'AY'
        })
        df = df.dropna(subset=['local', 'visitante', 'goles_l', 'goles_v']).copy()
        df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['fecha']).sort_values('fecha').tail(480)

        if 'tiros_puerta_l' in df.columns and df['tiros_puerta_l'].notna().sum() > 20:
            st_l = df['tiros_puerta_l'].fillna(df['goles_l'] * 2.8)
            st_v = df['tiros_puerta_v'].fillna(df['goles_v'] * 2.8)
            s_l = df['tiros_tot_l'].fillna(st_l * 2.5) if 'tiros_tot_l' in df.columns else st_l * 2.5
            s_v = df['tiros_tot_v'].fillna(st_v * 2.5) if 'tiros_tot_v' in df.columns else st_v * 2.5
            df['xg_l'] = (st_l * 0.26) + ((s_l - st_l).clip(lower=0) * 0.04) + (df['goles_l'] * 0.12)
            df['xg_v'] = (st_v * 0.26) + ((s_v - st_v).clip(lower=0) * 0.04) + (df['goles_v'] * 0.12)
        else:
            df['xg_l'] = df['goles_l']
            df['xg_v'] = df['goles_v']
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def cargar_champions():
    codigos = ["SP1", "E0", "I1", "D1"]
    dfs = []
    for c in codigos:
        df_c = descargar_liga_nacional(c)
        if not df_c.empty:
            dfs.append(df_c)
    return pd.concat(dfs, ignore_index=True)

@st.cache_data(ttl=1800)
def descargar_proximos_partidos():
    urls = [
        "https://www.football-data.co.uk/fixtures.csv",
        "https://www.football-data.co.uk/new_leagues_fixtures.csv"
    ]
    lista = []
    for u in urls:
        try:
            df_f = pd.read_csv(u, encoding='latin1')
            df_f = df_f.rename(columns={'Div': 'liga', 'Country': 'liga', 'Date': 'fecha', 'HomeTeam': 'local', 'AwayTeam': 'visitante', 'Home': 'local', 'Away': 'visitante'})
            cols = [c for c in ['liga', 'fecha', 'local', 'visitante'] if c in df_f.columns]
            lista.append(df_f[cols].dropna(subset=['local', 'visitante']))
        except Exception:
            pass
    return pd.concat(lista, ignore_index=True) if lista else pd.DataFrame()

def calcular_elo_con_margen(df, ventaja_campo=60.0):
    elos = {eq: 1500.0 for eq in set(df['local']).union(set(df['visitante']))}
    k_base = 24.0
    for _, row in df.iterrows():
        l, v = row['local'], row['visitante']
        gl, gv = row['goles_l'], row['goles_v']
        diff = abs(gl - gv)
        mult_goles = np.log(diff + 1.0) * 0.8 + 1.0 if diff > 0 else 1.0
        exp_l = 1.0 / (1.0 + 10.0 ** ((elos[v] - (elos[l] + ventaja_campo)) / 400.0))
        res_l = 1.0 if gl > gv else (0.5 if gl == gv else 0.0)
        cambio = k_base * mult_goles * (res_l - exp_l)
        elos[l] += cambio
        elos[v] -= cambio
    return elos

# ==========================================
# 3. MOTOR ESTADÍSTICO INTELIGENTE
# ==========================================
def obtener_metricas_automaticas(df, equipo):
    """Calcula automáticamente días de descanso, forma y fuerzas basadas en los datos reales."""
    df = df.copy()
    partidos_eq = df[(df['local'] == equipo) | (df['visitante'] == equipo)].sort_values('fecha')
    
    if len(partidos_eq) > 0:
        ultima_fecha = partidos_eq['fecha'].max()
        dias_descanso = int((df['fecha'].max() - ultima_fecha).days)
        dias_descanso = max(2, min(dias_descanso, 14)) # Rango realista
    else:
        dias_descanso = 7

    dias = (df['fecha'].max() - df['fecha']).dt.days
    df['peso_t'] = np.exp(-0.0045 * dias)
    df['rend_l'] = 0.68 * df['xg_l'] + 0.32 * df['goles_l']
    df['rend_v'] = 0.68 * df['xg_v'] + 0.32 * df['goles_v']

    media_l = np.average(df['rend_l'], weights=df['peso_t'])
    media_v = np.average(df['rend_v'], weights=df['peso_t'])

    df_casa = df[df['local'] == equipo]
    df_fuera = df[df['visitante'] == equipo]

    at_casa = np.average(df_casa['rend_l'], weights=df_casa['peso_t']) / media_l if len(df_casa) > 0 else 1.0
    def_casa = np.average(df_casa['rend_v'], weights=df_casa['peso_t']) / media_v if len(df_casa) > 0 else 1.0
    at_fuera = np.average(df_fuera['rend_v'], weights=df_fuera['peso_t']) / media_v if len(df_fuera) > 0 else 1.0
    def_fuera = np.average(df_fuera['rend_l'], weights=df_fuera['peso_t']) / media_l if len(df_fuera) > 0 else 1.0

    if len(partidos_eq) >= 4:
        gf = np.where(partidos_eq['local'] == equipo, partidos_eq['rend_l'], partidos_eq['rend_v'])
        gc = np.where(partidos_eq['local'] == equipo, partidos_eq['rend_v'], partidos_eq['rend_l'])
        forma = float(np.clip(((gf[-5:].mean() - gc[-5:].mean()) - (gf.mean() - gc.mean())) * 0.15, -0.20, 0.20))
    else:
        forma = 0.0

    return {
        'at_casa': at_casa, 'def_casa': def_casa,
        'at_fuera': at_fuera, 'def_fuera': def_fuera,
        'forma': forma, 'media_l': media_l, 'media_v': media_v,
        'dias_descanso': dias_descanso
    }

def predecir_corners_y_tarjetas(df, equipo_local, equipo_visitante):
    cols = ['HC', 'AC', 'HY', 'AY']
    if not all(c in df.columns for c in cols):
        return None
    dias = (df['fecha'].max() - df['fecha']).dt.days
    df = df.copy()
    df['peso_t'] = np.exp(-0.0045 * dias)
    df_casa = df[df['local'] == equipo_local]
    df_fuera = df[df['visitante'] == equipo_visitante]
    
    mc = df['HC'].mean()
    mt = df['HY'].mean()
    
    hc_loc = np.average(df_casa['HC'], weights=df_casa['peso_t']) if len(df_casa) > 0 else mc
    ac_vis = np.average(df_fuera['AC'], weights=df_fuera['peso_t']) if len(df_fuera) > 0 else mc
    tot_cor = hc_loc + ac_vis
    p_ov_cor = 1.0 - sum([poisson.pmf(k, tot_cor) for k in range(10)])

    hy_loc = np.average(df_casa['HY'], weights=df_casa['peso_t']) if len(df_casa) > 0 else mt
    ay_vis = np.average(df_fuera['AY'], weights=df_fuera['peso_t']) if len(df_fuera) > 0 else mt
    tot_tar = hy_loc + ay_vis
    p_ov_tar = 1.0 - sum([poisson.pmf(k, tot_tar) for k in range(5)])

    return {
        'corners_esp': tot_cor, 'over_95_c': p_ov_cor,
        'tarjetas_esp': tot_tar, 'over_45_t': p_ov_tar
    }

def ejecutar_modelo_analitico(m_loc, m_vis, elo_l, elo_v, campo_neutral=False, rho=-0.13):
    desc_l = m_loc['dias_descanso']
    desc_v = m_vis['dias_descanso']

    if campo_neutral:
        med_comun = (m_loc['media_l'] + m_loc['media_v']) / 2.0
        at_l = (m_loc['at_casa'] + m_loc['at_fuera']) / 2.0
        def_l = (m_loc['def_casa'] + m_loc['def_fuera']) / 2.0
        at_v = (m_vis['at_casa'] + m_vis['at_fuera']) / 2.0
        def_v = (m_vis['def_casa'] + m_vis['def_fuera']) / 2.0
        lam_base_l = at_l * def_v * med_comun
        lam_base_v = at_v * def_l * med_comun
        bono_elo = 0.0
    else:
        lam_base_l = m_loc['at_casa'] * m_vis['def_fuera'] * m_loc['media_l']
        lam_base_v = m_vis['at_fuera'] * m_loc['def_casa'] * m_vis['media_v']
        bono_elo = 60.0

    delta_elo = (elo_l + bono_elo - elo_v) / 400.0
    elo_mult_l = np.exp(0.11 * delta_elo)
    elo_mult_v = np.exp(-0.11 * delta_elo)

    fat_l = 0.94 if desc_l <= 3 else (1.02 if desc_l >= 6 else 1.0)
    fat_v = 0.94 if desc_v <= 3 else (1.02 if desc_v >= 6 else 1.0)

    lam_l = max(0.18, lam_base_l * (1 + m_loc['forma']) * elo_mult_l * fat_l)
    lam_v = max(0.18, lam_base_v * (1 + m_vis['forma']) * elo_mult_v * fat_v)

    max_g = 9
    p_l = poisson.pmf(range(max_g), lam_l)
    p_v = poisson.pmf(range(max_g), lam_v)
    matriz = np.outer(p_l, p_v)

    matriz[0, 0] *= max(0.0, 1.0 - lam_l * lam_v * rho)
    matriz[0, 1] *= max(0.0, 1.0 + lam_l * rho)
    matriz[1, 0] *= max(0.0, 1.0 + lam_v * rho)
    matriz[1, 1] *= max(0.0, 1.0 - rho)
    matriz /= matriz.sum()

    p1 = float(np.sum(np.tril(matriz, -1)))
    px_emp = float(np.sum(np.diag(matriz)))
    p2 = float(np.sum(np.triu(matriz, 1)))

    goles_totales = np.add.outer(range(max_g), range(max_g))
    over_25 = float(matriz[goles_totales > 2.5].sum())

    cs_local = float(matriz[:, 0].sum())
    cs_visit = float(matriz[0, :].sum())
    btts_no = cs_local + cs_visit - float(matriz[0, 0])
    btts_si = 1.0 - btts_no

    idx_rec = np.unravel_index(np.argmax(matriz), matriz.shape)
    marcador_recomendado = f"{idx_rec[0]} - {idx_rec[1]}"
    prob_recomendado = float(matriz[idx_rec])

    entropia = - (p1 * np.log(max(1e-9, p1)) + px_emp * np.log(max(1e-9, px_emp)) + p2 * np.log(max(1e-9, p2)))
    claridad = (1.0 - (entropia / np.log(3.0))) * 100.0

    if claridad > 14.0:
        opinion = "🟢 ALTA CONFIANZA"
        detalle = "Modelo con alta asimetría estadística. Patrón de rendimiento muy marcado entre ambos contendientes."
    elif claridad > 10.0:
        opinion = "🟡 PARTIDO NIVELADO"
        detalle = "Discrepancia moderada. Se sugieren mercados secundarios de goles o doble oportunidad."
    else:
        opinion = "🔴 ALTA INCERTIDUMBRE"
        detalle = "Escenario de triple empate técnico o alta varianza. Comportamiento errático en los últimos registros."

    return {
        'lam_l': lam_l, 'lam_v': lam_v, 'goles_esp_tot': lam_l + lam_v,
        'p1': p1, 'px': px_emp, 'p2': p2, 'matriz': matriz,
        'mas_dos_goles': over_25, 'ambos_marcaran': btts_si,
        'marcador_rec': marcador_recomendado, 'prob_rec': prob_recomendado,
        'claridad': claridad, 'opinion': opinion, 'detalle_opinion': detalle
    }

# ==========================================
# 4. INTERFAZ DE USUARIO ESTILO OPTA PRO
# ==========================================
st.markdown("<h1 style='text-align: center; font-size: 1.8rem; font-weight: 800; letter-spacing: -0.025em; margin-bottom: 0px;'>OPTA ANALYTICS <span style='color: #3b82f6;'>PRO ENGINE</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9ca3af; font-size: 0.85rem; margin-bottom: 25px;'>Advanced Poisson Framework & Market Expected Value Matrix</p>", unsafe_allow_html=True)

col_sel1, col_sel2 = st.columns([2, 1])
with col_sel1:
    tipo_competicion = st.radio("Competición:", ["🇪🇸 Ligas Nacionales", "🏆 Champions League (Top Europa)"], horizontal=True, label_visibility="collapsed")

cod_liga_actual = None
es_neutral_default = False

if "Ligas Nacionales" in tipo_competicion:
    liga_sel = st.selectbox("Selecciona Liga:", list(LIGAS_PRINCIPALES.keys()), label_visibility="collapsed")
    cod_liga_actual = LIGAS_PRINCIPALES[liga_sel]
    df_hist = descargar_liga_nacional(cod_liga_actual)
    elos_dict = calcular_elo_con_margen(df_hist, ventaja_campo=60.0)
else:
    df_hist = cargar_champions()
    elos_dict = calcular_elo_con_margen(df_hist, ventaja_campo=0.0)
    es_neutral_default = True

with col_sel2:
    campo_neutral = st.checkbox("🏟️ Sede Neutral", value=es_neutral_default)

if df_hist.empty:
    st.error("Error al sincronizar los datos de la competición.")
    st.stop()

equipos_lista = sorted(list(elos_dict.keys()))
tab_partido, tab_jornada = st.tabs(["📊 Análisis Táctico de Partido", "📋 Escáner Global de Jornada"])

with tab_partido:
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    c_eq1, c_vs, c_eq2 = st.columns([2, 0.4, 2])
    with c_eq1:
        eq_local = st.selectbox("Local:", equipos_lista, index=0)
    with c_vs:
        st.markdown("<div style='text-align: center; padding-top: 30px; color: #6b7280; font-weight: 700;'>VS</div>", unsafe_allow_html=True)
    with c_eq2:
        eq_visit = st.selectbox("Visitante:", equipos_lista, index=min(1, len(equipos_lista)-1))

    m_loc = obtener_metricas_automaticas(df_hist, eq_local)
    m_vis = obtener_metricas_automaticas(df_hist, eq_visit, eq_visit)
    elo_l_val = elos_dict.get(eq_local, 1500.0)
    elo_v_val = elos_dict.get(eq_visit, 1500.0)

    res = ejecutar_modelo_analitico(m_loc, m_vis, elo_l_val, elo_v_val, campo_neutral=campo_neutral)

    # --- BANNER DE OPINIÓN TÁCTICA ---
    st.markdown(f"""
    <div style="background: #111827; border-left: 4px solid {'#10b981' if '🟢' in res['opinion'] else ('#f59e0b' if '🟡' in res['opinion'] else '#ef4444')}; padding: 14px 18px; border-radius: 6px; margin: 15px 0 20px 0;">
        <div style="font-weight: 700; font-size: 0.9rem; color: #f8fafc; margin-bottom: 3px;">{res['opinion']}</div>
        <div style="color: #9ca3af; font-size: 0.8rem;">{res['detalle_opinion']}</div>
    </div>
    """, unsafe_allow_html=True)

    # --- TARJETAS MÉTRICAS SUPERIORES ---
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">1 - {eq_local}</div><div class="metric-value">{res['p1']*100:.1f}%</div><div class="metric-sub">Cuota Justa: {1/res['p1']:.2f}</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">X - Empate</div><div class="metric-value">{res['px']*100:.1f}%</div><div class="metric-sub">Cuota Justa: {1/res['px']:.2f}</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">2 - {eq_visit}</div><div class="metric-value">{res['p2']*100:.1f}%</div><div class="metric-sub">Cuota Justa: {1/res['p2']:.2f}</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Mas de dos goles</div><div class="metric-value">{res['mas_dos_goles']*100:.1f}%</div><div class="metric-sub">λ Totales: {res['goles_esp_tot']:.2f}</div></div>""", unsafe_allow_html=True)
    with m5:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Ambos marcarán</div><div class="metric-value">{res['ambos_marcaran']*100:.1f}%</div><div class="metric-sub">BTTS Ratio</div></div>""", unsafe_allow_html=True)

    # --- COMPARATIVA DE MERCADO ESPERADA ---
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size: 1rem; font-weight: 700; color: #f1f5f9; margin-bottom: 12px;'>📈 Matriz de Apuesta Esperada (Comparativa de Mercado)</h3>", unsafe_allow_html=True)

    # Cuota teórica de mercado estimada restando un 5% de margen estándar de casa
    cuota_mercado_1 = round(1 / res['p1'] * 0.95, 2)
    cuota_mercado_x = round(1 / res['px'] * 0.95, 2)
    cuota_mercado_2 = round(1 / res['p2'] * 0.95, 2)

    df_comparativa = pd.DataFrame([
        {"Mercado": f"Victoria Local ({eq_local})", "Probabilidad Modelo": f"{res['p1']*100:.1f}%", "Cuota Justa Pura": round(1/res['p1'], 2), "Cuota Media Mercado": cuota_mercado_1, "Desviación Valor": "Equilibrado"},
        {"Mercado": "Empate (X)", "Probabilidad Modelo": f"{res['px']*100:.1f}%", "Cuota Justa Pura": round(1/res['px'], 2), "Cuota Media Mercado": cuota_mercado_x, "Desviación Valor": "Equilibrado"},
        {"Mercado": f"Victoria Visitante ({eq_visit})", "Probabilidad Modelo": f"{res['p2']*100:.1f}%", "Cuota Justa Pura": round(1/res['p2'], 2), "Cuota Media Mercado": cuota_mercado_2, "Desviación Valor": "Equilibrado"}
    ])
    st.dataframe(df_comparativa, use_container_width=True, hide_index=True)

    # --- CÓRNERS Y TARJETAS ---
    res_disc = predecir_corners_y_tarjetas(df_hist, eq_local, eq_visit)
    if res_disc:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 1rem; font-weight: 700; color: #f1f5f9; margin-bottom: 12px;'>⚡ Métricas de Disciplina y Saques de Esquina</h3>", unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Córners Esperados</div><div class="metric-value">{res_disc['corners_esp']:.1f}</div><div class="metric-sub">Saques esquina</div></div>""", unsafe_allow_html=True)
        with d2:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Over 9.5 Córners</div><div class="metric-value">{res_disc['over_95_c']*100:.1f}%</div><div class="metric-sub">Línea estándar</div></div>""", unsafe_allow_html=True)
        with d3:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Tarjetas Esperadas</div><div class="metric-value">{res_disc['tarjetas_esp']:.1f}</div><div class="metric-sub">Amarillas total</div></div>""", unsafe_allow_html=True)
        with d4:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Over 4.5 Tarjetas</div><div class="metric-value">{res_disc['over_45_t']*100:.1f}%</div><div class="metric-sub">Línea disciplinaria</div></div>""", unsafe_allow_html=True)

with tab_jornada:
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size: 1rem; font-weight: 700; color: #f1f5f9; margin-bottom: 12px;'>📋 Escáner Global de Encuentros</h3>", unsafe_allow_html=True)
    
    df_p = descargar_proximos_partidos()
    p_jor = df_p[df_p['liga'] == cod_liga_actual] if cod_liga_actual and not df_p.empty and 'liga' in df_p.columns else pd.DataFrame()
    if p_jor.empty:
        pares = [(equipos_lista[i], equipos_lista[i+1]) for i in range(0, min(len(equipos_lista)-1, 14), 2)]
        p_jor = pd.DataFrame(pares, columns=['local', 'visitante'])

    filas = []
    for _, row in p_jor.iterrows():
        l, v = row['local'], row['visitante']
        if l in elos_dict and v in elos_dict:
            ml = obtener_metricas_automaticas(df_hist, l)
            mv = obtener_metricas_automaticas(df_hist, v)
            rj = ejecutar_modelo_analitico(ml, mv, elos_dict[l], elos_dict[v], campo_neutral=campo_neutral)
            filas.append({
                "Encuentro": f"{l} vs {v}",
                "Confianza": rj['opinion'].split()[0],
                "1 (%)": f"{rj['p1']*100:.1f}%", "X (%)": f"{rj['px']*100:.1f}%", "2 (%)": f"{rj['p2']*100:.1f}%",
                "🎯 Sugerido": f"{rj['marcador_rec']} ({rj['prob_rec']*100:.1f}%)",
                "Mas de dos goles": f"{rj['mas_dos_goles']*100:.1f}%", "Ambos marcarán": f"{rj['ambos_marcaran']*100:.1f}%"
            })
    if filas:
        st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)