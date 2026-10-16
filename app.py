import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import altair as alt
from datetime import datetime, timedelta

# Configuración inicial de la página
st.set_page_config(
    page_title="Opta Analytics | Pro Engine", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# ESTILOS CSS - LENGUAJE VISUAL DE APPLE (MODO PRO)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #000000;
        color: #f5f5f7;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Inter", sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .metric-card {
        background: rgba(28, 28, 30, 0.75);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(0, 113, 227, 0.6);
        transform: scale(1.01);
    }
    .metric-label {
        color: #86868b;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: -0.01em;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #f5f5f7;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.03em;
    }
    .metric-sub {
        color: #2997ff;
        font-size: 0.75rem;
        margin-top: 4px;
        font-weight: 500;
    }

    .stSelectbox label, .stRadio label {
        color: #86868b !important;
        font-weight: 600;
        font-size: 0.8rem;
        letter-spacing: -0.01em;
    }
    
    div[data-baseweb="select"] > div {
        background-color: rgba(28, 28, 30, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        color: #f5f5f7 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. CATÁLOGO DE LIGAS INDIVIDUALES
# ==========================================
LIGAS_MAPA = {
    "🇪🇸 España - LaLiga EA Sports (1ª)": "SP1",
    "🇪🇸 España - LaLiga Hypermotion (2ª)": "SP2",
    "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Inglaterra - Premier League": "E0",
    "🇮🇹 Italia - Serie A": "I1",
    "🇩🇪 Alemania - Bundesliga": "D1",
    "🇫🇷 Francia - Ligue 1": "F1",
    "🇵🇹 Portugal - Primeira Liga": "P1"
}

LIGAS_OFICIALES = {
    "🌍 Global - Partidos Cruzados por Ligas": "GLOBAL",
    **LIGAS_MAPA,
    "🏆 UEFA Champions League": "UCL"
}

CALENDARIOS_OFICIALES = {
    "SP1": {
        "Jornada 9 (16-19 Oct 2026)": [
            ("Deportivo", "Levante"), ("Espanyol", "Atletico Madrid"), ("Villarreal", "Elche"),
            ("Real Betis", "Barcelona"), ("Valencia", "Athletic Club"), ("Osasuna", "Racing Santander"),
            ("Celta", "Alaves"), ("Malaga", "Real Sociedad"), ("Real Madrid", "Sevilla"), ("Getafe", "Rayo Vallecano")
        ],
        "Jornada 10 (23-26 Oct 2026)": [
            ("Alaves", "Malaga"), ("Rayo Vallecano", "Elche"), ("Racing Santander", "Espanyol"),
            ("Valencia", "Villarreal"), ("Atletico Madrid", "Deportivo"), ("Athletic Club", "Getafe"),
            ("Celta", "Real Betis"), ("Real Sociedad", "Levante"), ("Barcelona", "Real Madrid"), ("Sevilla", "Osasuna")
        ]
    }
}

# ==========================================
# 2. CONECTOR DE DATOS UNIFICADO
# ==========================================
@st.cache_data(ttl=3600)
def descargar_datos_liga(codigo_liga, temp_act="2627", temp_prev="2526"):
    try:
        codigos = list(LIGAS_MAPA.values()) if codigo_liga in ["UCL", "GLOBAL"] else [codigo_liga]
        dfs = []
        for c in codigos:
            for t in [temp_prev, temp_act]:
                url = f"https://www.football-data.co.uk/mmz4281/{t}/{c}.csv"
                try:
                    dfs.append(pd.read_csv(url, encoding='latin1'))
                except Exception:
                    pass
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        df = df.rename(columns={
            'Date': 'fecha', 'HomeTeam': 'local', 'AwayTeam': 'visitante',
            'FTHG': 'goles_l', 'FTAG': 'goles_v', 'HST': 'tiros_puerta_l', 'AST': 'tiros_puerta_v',
            'HS': 'tiros_tot_l', 'AS': 'tiros_tot_v', 'HC': 'HC', 'AC': 'AC', 'HY': 'HY', 'AY': 'AY'
        })
        df = df.dropna(subset=['local', 'visitante', 'goles_l', 'goles_v']).copy()
        df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['fecha']).sort_values('fecha')

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
def descargar_datos_individuales(codigo_liga):
    try:
        dfs = []
        for t in ["2526", "2627"]:
            url = f"https://www.football-data.co.uk/mmz4281/{t}/{codigo_liga}.csv"
            try:
                dfs.append(pd.read_csv(url, encoding='latin1'))
            except Exception:
                pass
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        if df.empty:
            return set()
        df = df.rename(columns={'HomeTeam': 'local', 'AwayTeam': 'visitante'})
        return sorted(list(set(df['local']).union(set(df['visitante']))))
    except Exception:
        return []

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
# 3. MOTOR ESTADÍSTICO Y RECOMENDACIÓN
# ==========================================
def obtener_metricas_automaticas(df, equipo):
    df = df.copy()
    df['rend_l'] = 0.68 * df['xg_l'] + 0.32 * df['goles_l']
    df['rend_v'] = 0.68 * df['xg_v'] + 0.32 * df['goles_v']

    dias = (df['fecha'].max() - df['fecha']).dt.days
    df['peso_t'] = np.exp(-0.0045 * dias)

    media_l = np.average(df['rend_l'], weights=df['peso_t'])
    media_v = np.average(df['rend_v'], weights=df['peso_t'])

    partidos_eq = df[(df['local'] == equipo) | (df['visitante'] == equipo)].sort_values('fecha')
    
    if len(partidos_eq) > 0:
        ultima_fecha = partidos_eq['fecha'].max()
        dias_descanso = int((df['fecha'].max() - ultima_fecha).days)
        dias_descanso = max(2, min(dias_descanso, 14))
    else:
        dias_descanso = 7

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

def ejecutar_modelo_analitico(m_loc, m_vis, elo_l, elo_v, eq_local, eq_visit, campo_neutral=False, rho=-0.13):
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

    if p1 > px_emp and p1 > p2:
        quiniela = "1"
    elif p2 > p1 and p2 > px_emp:
        quiniela = "2"
    else:
        quiniela = "X"

    entropia = - (p1 * np.log(max(1e-9, p1)) + px_emp * np.log(max(1e-9, px_emp)) + p2 * np.log(max(1e-9, p2)))
    claridad = (1.0 - (entropia / np.log(3.0))) * 100.0

    if claridad > 14.0:
        opinion = "🟢 ALTA CONFIANZA"
        detalle = "Modelo con alta asimetría estadística."
    elif claridad > 10.0:
        opinion = "🟡 PARTIDO NIVELADO"
        detalle = "Discrepancia moderada."
    else:
        opinion = "🔴 ALTA INCERTIDUMBRE"
        detalle = "Escenario de alta varianza."

    max_p = max(p1, px_emp, p2)
    if max_p == p1:
        apuesta_val = f"Victoria {eq_local} (1)" if p1 > 0.45 else f"1X (Doble Oportunidad)"
    elif max_p == p2:
        apuesta_val = f"Victoria {eq_visit} (2)" if p2 > 0.45 else f"X2 (Doble Oportunidad)"
    else:
        apuesta_val = "Empate (X) o Menos 2.5 Goles"

    if over_25 > 0.58:
        apuesta_val += " + Más 2.5 Goles"
    elif over_25 < 0.40:
        apuesta_val += " + Menos 3.5 Goles"

    return {
        'lam_l': lam_l, 'lam_v': lam_v, 'goles_esp_tot': lam_l + lam_v,
        'p1': p1, 'px': px_emp, 'p2': p2, 'quiniela': quiniela,
        'mas_dos_goles': over_25, 'ambos_marcaran': btts_si,
        'marcador_rec': marcador_recomendado, 'prob_rec': prob_recomendado,
        'opinion': opinion, 'detalle_opinion': detalle, 'apuesta_val': apuesta_val
    }

# ==========================================
# 4. INTERFAZ PRINCIPAL
# ==========================================
st.markdown("<h1 style='text-align: center; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.025em; margin-bottom: 0px;'>Opta Analytics <span style='color: #0071e3;'>Pro</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #86868b; font-size: 0.85rem; margin-bottom: 25px;'>Advanced Bayesian Engine & Live Market Value</p>", unsafe_allow_html=True)

modo = st.radio(
    "Modo de Trabajo:",
    ["📊 Análisis Táctico Libre (Cualquier Equipo)", "📅 Módulo de Jornada Oficial"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

col_l1, col_l2 = st.columns([2, 1])
with col_l1:
    liga_sel = st.selectbox("Selecciona Competición / Modo Global:", list(LIGAS_OFICIALES.keys()))
    cod_liga_actual = LIGAS_OFICIALES[liga_sel]

df_hist = descargar_datos_liga(cod_liga_actual)
elos_dict = calcular_elo_con_margen(df_hist, ventaja_campo=0.0 if cod_liga_actual in ["UCL", "GLOBAL"] else 60.0)

with col_l2:
    campo_neutral = st.checkbox("🏟️ Sede Neutral", value=(cod_liga_actual in ["UCL", "GLOBAL"]))

if df_hist.empty:
    st.error("Error al sincronizar los datos de la competición.")
    st.stop()

# ==========================================
# MÓDULO 1: ANÁLISIS TÁCTICO LIBRE (CON SUBMENÚS DE LIGAS)
# ==========================================
if modo == "📊 Análisis Táctico Libre (Cualquier Equipo)":
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size: 1.1rem; font-weight: 600; color: #f5f5f7; margin-bottom: 15px;'>⚡ Enfrentamiento Directo y Toma de Decisiones</h3>", unsafe_allow_html=True)
    
    if cod_liga_actual == "GLOBAL":
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            liga_local_sel = st.selectbox("Liga del Equipo Local:", list(LIGAS_MAPA.keys()))
            equipos_local_lista = descargar_datos_individuales(LIGAS_MAPA[liga_local_sel])
        with c_sub2:
            liga_visit_sel = st.selectbox("Liga del Equipo Visitante:", list(LIGAS_MAPA.keys()), index=min(1, len(LIGAS_MAPA)-1))
            equipos_visit_lista = descargar_datos_individuales(LIGAS_MAPA[liga_visit_sel])

        c_eq1, c_vs, c_eq2 = st.columns([2, 0.4, 2])
        with c_eq1:
            eq_local = st.selectbox("Equipo Local:", equipos_local_lista if equipos_local_lista else list(elos_dict.keys()), index=0)
        with c_vs:
            st.markdown("<div style='text-align: center; padding-top: 30px; color: #86868b; font-weight: 600;'>VS</div>", unsafe_allow_html=True)
        with c_eq2:
            eq_visit = st.selectbox("Equipo Visitante:", equipos_visit_lista if equipos_visit_lista else list(elos_dict.keys()), index=min(1, len(equipos_visit_lista)-1))
    else:
        equipos_lista = sorted(list(elos_dict.keys()))
        c_eq1, c_vs, c_eq2 = st.columns([2, 0.4, 2])
        with c_eq1:
            eq_local = st.selectbox("Equipo Local:", equipos_lista, index=0)
        with c_vs:
            st.markdown("<div style='text-align: center; padding-top: 30px; color: #86868b; font-weight: 600;'>VS</div>", unsafe_allow_html=True)
        with c_eq2:
            eq_visit = st.selectbox("Equipo Visitante:", equipos_lista, index=min(1, len(equipos_lista)-1))

    m_loc = obtener_metricas_automaticas(df_hist, eq_local)
    m_vis = obtener_metricas_automaticas(df_hist, eq_visit)
    elo_l_val = elos_dict.get(eq_local, 1500.0)
    elo_v_val = elos_dict.get(eq_visit, 1500.0)

    res = ejecutar_modelo_analitico(m_loc, m_vis, elo_l_val, elo_v_val, eq_local, eq_visit, campo_neutral=campo_neutral)

    # Banner de Opinión
    st.markdown(f"""
    <div style="background: rgba(28, 28, 30, 0.6); backdrop-filter: blur(10px); border-left: 4px solid {'#34c759' if '🟢' in res['opinion'] else ('#ff9f0a' if '🟡' in res['opinion'] else '#ff453a')}; padding: 14px 18px; border-radius: 12px; margin: 20px 0;">
        <div style="font-weight: 600; font-size: 0.9rem; color: #f5f5f7; margin-bottom: 3px;">{res['opinion']}</div>
        <div style="color: #86868b; font-size: 0.8rem;">{res['detalle_opinion']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Tarjetas de Métricas Principales
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">1 - {eq_local}</div><div class="metric-value">{res['p1']*100:.1f}%</div><div class="metric-sub">Cuota: {1/res['p1']:.2f}</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">X - Empate</div><div class="metric-value">{res['px']*100:.1f}%</div><div class="metric-sub">Cuota: {1/res['px']:.2f}</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">2 - {eq_visit}</div><div class="metric-value">{res['p2']*100:.1f}%</div><div class="metric-sub">Cuota: {1/res['p2']:.2f}</div></div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Mas de dos goles</div><div class="metric-value">{res['mas_dos_goles']*100:.1f}%</div><div class="metric-sub">λ: {res['goles_esp_tot']:.2f}</div></div>""", unsafe_allow_html=True)
    with m5:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Ambos marcarán</div><div class="metric-value">{res['ambos_marcaran']*100:.1f}%</div><div class="metric-sub">BTTS Ratio</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
    st.markdown("<h4 style='font-size: 0.95rem; font-weight: 600; color: #f5f5f7; margin-bottom: 12px;'>📋 Resumen Ejecutivo y Recomendación de Apuesta</h4>", unsafe_allow_html=True)
    
    df_resumen_individual = pd.DataFrame([{
        "Encuentro": f"{eq_local} vs {eq_visit}",
        "Quiniela (1X2)": res['quiniela'],
        "Confianza": res['opinion'].split()[0],
        "1 (%)": f"{res['p1']*100:.1f}%", 
        "X (%)": f"{res['px']*100:.1f}%", 
        "2 (%)": f"{res['p2']*100:.1f}%",
        "🎯 Sugerido": f"{res['marcador_rec']} ({res['prob_rec']*100:.1f}%)",
        "💡 Qué Apostar": res['apuesta_val']
    }])
    st.dataframe(df_resumen_individual, use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
    st.markdown("<h4 style='font-size: 0.95rem; font-weight: 600; color: #f5f5f7; margin-bottom: 12px;'>📈 Gráficos de Apoyo y Factores Clave para Decisiones</h4>", unsafe_allow_html=True)

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("<p style='font-size: 0.8rem; color: #86868b; font-weight: 600;'>Distribución de Goles Esperados (λ Local vs Visitante)</p>", unsafe_allow_html=True)
        goles_df = pd.DataFrame({
            'Goles': list(range(5)) * 2,
            'Probabilidad': list(poisson.pmf(range(5), res['lam_l'])) + list(poisson.pmf(range(5), res['lam_v'])),
            'Equipo': [eq_local]*5 + [eq_visit]*5
        })
        chart_goles = alt.Chart(goles_df).mark_bar(opacity=0.8, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X('Goles:O', title='Número de Goles'),
            y=alt.Y('Probabilidad:Q', title='Probabilidad'),
            color=alt.Color('Equipo:N', scale=alt.Scale(range=['#0071e3', '#34c759'])),
            column='Equipo:N'
        ).properties(width=220, height=180).configure_view(stroke=None)
        st.altair_chart(chart_goles, use_container_width=True)

    with col_g2:
        st.markdown("<p style='font-size: 0.8rem; color: #86868b; font-weight: 600;'>Métricas Tácticas y Descanso (Ayuda de Decisión)</p>", unsafe_allow_html=True)
        metrics_df = pd.DataFrame({
            'Factor': ['Ataque Casa/Fuera', 'Defensa (Inv.)', 'Días Descanso', 'Forma Reciente'],
            eq_local: [m_loc['at_casa'], 2.0 - m_loc['def_casa'], m_loc['dias_descanso']/14.0, 1.0 + m_loc['forma']],
            eq_visit: [m_vis['at_fuera'], 2.0 - m_vis['def_fuera'], m_vis['dias_descanso']/14.0, 1.0 + m_vis['forma']]
        }).melt('Factor', var_name='Equipo', value_name='Score')

        chart_radar = alt.Chart(metrics_df).mark_bar().encode(
            x=alt.X('Score:Q', title='Índice Normalizado'),
            y=alt.Y('Factor:N', sort=None, title=''),
            color=alt.Color('Equipo:N', scale=alt.Scale(range=['#0071e3', '#ff9f0a']))
        ).properties(width=300, height=180).configure_view(stroke=None)
        st.altair_chart(chart_radar, use_container_width=True)

# ==========================================
# MÓDULO 2: MÓDULO DE JORNADA OFICIAL
# ==========================================
else:
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='font-size: 1.1rem; font-weight: 600; color: #f5f5f7; margin-bottom: 15px;'>📅 Cuadro Resumen de Jornada — {liga_sel}</h3>", unsafe_allow_html=True)
    
    calendario_liga = CALENDARIOS_OFICIALES.get(cod_liga_actual, {})
    
    if calendario_liga:
        jornada_elegida = st.selectbox("Selecciona la Jornada Oficial:", list(calendario_liga.keys()))
        partidos_jornada = calendario_liga[jornada_elegida]
    else:
        st.info("Para esta competición, utiliza el modo de 'Análisis Táctico Libre' para enfrentar a cualquier equipo de la categoría.")
        partidos_jornada = []

    if partidos_jornada:
        filas_jornada = []
        for l, v in partidos_jornada:
            l_match = l if l in elos_dict else next((eq for eq in elos_dict if l.lower() in eq.lower()), l)
            v_match = v if v in elos_dict else next((eq for eq in elos_dict if v.lower() in eq.lower()), v)
            
            if l_match in elos_dict and v_match in elos_dict and l_match != v_match:
                ml = obtener_metricas_automaticas(df_hist, l_match)
                mv = obtener_metricas_automaticas(df_hist, v_match)
                rj = ejecutar_modelo_analitico(ml, mv, elos_dict[l_match], elos_dict[v_match], l_match, v_match, campo_neutral=False)
                
                filas_jornada.append({
                    "Encuentro": f"{l_match} vs {v_match}",
                    "Quiniela (1X2)": rj['quiniela'],
                    "Confianza": rj['opinion'].split()[0],
                    "1 (%)": f"{rj['p1']*100:.1f}%", 
                    "X (%)": f"{rj['px']*100:.1f}%", 
                    "2 (%)": f"{rj['p2']*100:.1f}%",
                    "🎯 Sugerido": f"{rj['marcador_rec']} ({rj['prob_rec']*100:.1f}%)",
                    "💡 Qué Apostar": rj['apuesta_val']
                })
        
        if filas_jornada:
            st.dataframe(pd.DataFrame(filas_jornada), use_container_width=True, hide_index=True)
        else:
            st.warning("No hay suficientes datos de encuentros cruzados para esta jornada.")