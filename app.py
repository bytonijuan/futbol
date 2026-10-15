import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configuración inicial de la página
st.set_page_config(
    page_title="Predictor Pro | Analytics", 
    page_icon="⚽", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# ESTILOS CSS PREMIUM (Modo Oscuro / Deportivo)
# ==========================================
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #f0f2f6;
    }
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }
    div[data-testid="stMetric"] label {
        color: #8b949e !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f0f6fc !important;
        font-size: 1.3rem !important;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #161b22;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #8b949e;
        border-radius: 6px;
        font-weight: 600;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #238636 !important;
        color: #ffffff !important;
    }
    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. CATÁLOGO DE COMPETICIONES
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
    """Carga los mejores equipos de Europa cruzando España, Inglaterra, Italia y Alemania para la Champions."""
    codigos = ["SP1", "E0", "I1", "D1"]
    dfs = []
    for c in codigos:
        df_c = descargar_liga_nacional(c)
        if not df_c.empty:
            dfs.append(df_c)
    df_tot = pd.concat(dfs, ignore_index=True)
    # Filtramos solo clubes de nivel alto o top para simular la Champions
    return df_tot

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
# 3. MOTOR ESTADÍSTICO
# ==========================================
def obtener_metricas_fiables(df, equipo, elos_dict):
    df = df.copy()
    dias = (df['fecha'].max() - df['fecha']).dt.days
    df['peso_t'] = np.exp(-0.0045 * dias)
    df['rend_l'] = 0.68 * df['xg_l'] + 0.32 * df['goles_l']
    df['rend_v'] = 0.68 * df['xg_v'] + 0.32 * df['goles_v']

    media_l = np.average(df['rend_l'], weights=df['peso_t'])
    media_v = np.average(df['rend_v'], weights=df['peso_t'])
    elo_medio_liga = np.mean(list(elos_dict.values())) if elos_dict else 1500.0

    df_casa = df[df['local'] == equipo]
    df_fuera = df[df['visitante'] == equipo]
    c_bayes = 6.0

    n_c = len(df_casa)
    if n_c > 0:
        ajuste_rival_c = np.array([elos_dict.get(r, 1500.0) / elo_medio_liga for r in df_casa['visitante']])
        at_c_raw = np.average(df_casa['rend_l'] * ajuste_rival_c, weights=df_casa['peso_t']) / media_l
        def_c_raw = np.average(df_casa['rend_v'] / ajuste_rival_c, weights=df_casa['peso_t']) / media_v
        at_casa = (n_c * at_c_raw + c_bayes * 1.0) / (n_c + c_bayes)
        def_casa = (n_c * def_c_raw + c_bayes * 1.0) / (n_c + c_bayes)
    else:
        at_casa, def_casa = 1.0, 1.0

    n_f = len(df_fuera)
    if n_f > 0:
        ajuste_rival_f = np.array([elos_dict.get(r, 1500.0) / elo_medio_liga for r in df_fuera['local']])
        at_f_raw = np.average(df_fuera['rend_v'] * ajuste_rival_f, weights=df_fuera['peso_t']) / media_v
        def_f_raw = np.average(df_fuera['rend_l'] / ajuste_rival_f, weights=df_fuera['peso_t']) / media_l
        at_fuera = (n_f * at_f_raw + c_bayes * 1.0) / (n_f + c_bayes)
        def_fuera = (n_f * def_f_raw + c_bayes * 1.0) / (n_f + c_bayes)
    else:
        at_fuera, def_fuera = 1.0, 1.0

    partidos_eq = df[(df['local'] == equipo) | (df['visitante'] == equipo)].sort_values('fecha')
    if len(partidos_eq) >= 4:
        gf = np.where(partidos_eq['local'] == equipo, partidos_eq['rend_l'], partidos_eq['rend_v'])
        gc = np.where(partidos_eq['local'] == equipo, partidos_eq['rend_v'], partidos_eq['rend_l'])
        dif_temp = (gf - gc).mean()
        dif_ult5 = (gf[-5:] - gc[-5:]).mean()
        forma = float(np.clip((dif_ult5 - dif_temp) * 0.18, -0.22, 0.22))
    else:
        forma = 0.0

    return {
        'at_casa': at_casa, 'def_casa': def_casa,
        'at_fuera': at_fuera, 'def_fuera': def_fuera,
        'forma': forma, 'media_l': media_l, 'media_v': media_v
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

def ejecutar_modelo_completo(m_loc, m_vis, elo_l, elo_v, forma_l, forma_v, bajas_l, bajas_v, desc_l, desc_v, cuota_1, cuota_x, cuota_2, campo_neutral=False, rho=-0.13):
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
        lam_base_v = m_vis['at_fuera'] * m_vis['def_casa'] * m_vis['media_v']
        bono_elo = 60.0

    delta_elo = (elo_l + bono_elo - elo_v) / 400.0
    elo_mult_l = np.exp(0.11 * delta_elo)
    elo_mult_v = np.exp(-0.11 * delta_elo)

    fat_l = 0.93 if desc_l <= 3 else (1.03 if desc_l >= 6 else 1.0)
    fat_v = 0.93 if desc_v <= 3 else (1.03 if desc_v >= 6 else 1.0)

    lam_l = max(0.18, lam_base_l * (1 + forma_l) * elo_mult_l * (1 - bajas_l) * fat_l)
    lam_v = max(0.18, lam_base_v * (1 + forma_v) * elo_mult_v * (1 - bajas_v) * fat_v)

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

    ev1 = (p1 * cuota_1) - 1.0
    evx = (px_emp * cuota_x) - 1.0
    ev2 = (p2 * cuota_2) - 1.0
    max_ev = max(ev1, evx, ev2)

    if claridad > 14.0 and max_ev > 0.04:
        opinion = "🟢 ALTO VALOR (Recomendado)"
        detalle = "El modelo detecta una clara asimetría competitiva combinada con cuotas ineficientes. Merece la pena buscar la opción con EV+."
    elif claridad > 10.0 and max_ev > 0.0:
        opinion = "🟡 MODERADO (Precaución)"
        detalle = "Hay ligera ventaja matemática, pero el partido tiene tramos de incertidumbre. Se aconseja reducir el stake o usar Doble Oportunidad."
    else:
        opinion = "🔴 EVITAR / PARTIDO TRAMPA"
        detalle = "Las probabilidades están muy equilibradas o las cuotas no compensan el riesgo. Es mejor no arriesgar capital en este encuentro."

    return {
        'lam_l': lam_l, 'lam_v': lam_v, 'goles_esp_tot': lam_l + lam_v,
        'p1': p1, 'px': px_emp, 'p2': p2, 'matriz': matriz,
        'mas_dos_goles': over_25, 'ambos_marcaran': btts_si,
        'marcador_rec': marcador_recomendado, 'prob_rec': prob_recomendado,
        'claridad': claridad, 'opinion': opinion, 'detalle_opinion': detalle
    }

def evaluar_ev_kelly(prob_modelo, cuota_casa, fraccion_kelly, bankroll):
    ev = (prob_modelo * cuota_casa) - 1.0
    if ev > 0 and cuota_casa > 1.0:
        kelly_puro = ev / (cuota_casa - 1.0)
        stake_pct = min(max(0.0, kelly_puro * fraccion_kelly), 0.10)
    else:
        stake_pct = 0.0
    return ev * 100.0, stake_pct * 100.0, bankroll * stake_pct

# ==========================================
# 4. INTERFAZ DE USUARIO PREMIUM
# ==========================================
st.markdown("<h2 style='text-align: center; color: #58a6ff;'>⚽ ELITE FOOTBALL ANALYTICS</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e;'>Modelo Bayesiano de Probabilidad & Valor Esperado (EV+)</p>", unsafe_allow_html=True)
st.markdown("---")

tipo_competicion = st.radio("Selecciona tipo de competición:", ["🇪🇸 Ligas Nacionales", "🏆 Champions League (Top Europa)"], horizontal=True)

cod_liga_actual = None
es_neutral_default = False

if "Ligas Nacionales" in tipo_competicion:
    liga_sel = st.selectbox("Selecciona Liga:", list(LIGAS_PRINCIPALES.keys()))
    cod_liga_actual = LIGAS_PRINCIPALES[liga_sel]
    df_hist = descargar_liga_nacional(cod_liga_actual)
    elos_dict = calcular_elo_con_margen(df_hist, ventaja_campo=60.0)
else:
    df_hist = cargar_champions()
    elos_dict = calcular_elo_con_margen(df_hist, ventaja_campo=0.0) # Champions suele ser nivelado o mixto
    es_neutral_default = True

campo_neutral = st.checkbox("🏟️ Sede Neutral (Campo neutral)", value=es_neutral_default)

if df_hist.empty:
    st.error("No se pudieron cargar los datos de esta competición.")
    st.stop()

equipos_lista = sorted(list(elos_dict.keys()))
tab_partido, tab_jornada = st.tabs(["🔍 Análisis de Partido & Kelly", "📋 Pronóstico de Jornada"])

with tab_partido:
    c_eq1, c_eq2 = st.columns(2)
    with c_eq1:
        eq_local = st.selectbox("Local:", equipos_lista, index=0)
    with c_eq2:
        eq_visit = st.selectbox("Visitante:", equipos_lista, index=min(1, len(equipos_lista)-1))

    m_loc = obtener_metricas_fiables(df_hist, eq_local, elos_dict)
    m_vis = obtener_metricas_fiables(df_hist, eq_visit, elos_dict)
    elo_l_val = elos_dict.get(eq_local, 1500.0)
    elo_v_val = elos_dict.get(eq_visit, 1500.0)

    st.sidebar.header("⚙️ Contexto Deportivo")
    st.sidebar.subheader(f"🏠 {eq_local} (Elo: {int(elo_l_val)})")
    forma_l_ui = st.sidebar.slider(f"Forma {eq_local} (%)", -25, 25, int(m_loc['forma']*100), 1) / 100.0
    bajas_l_ui = st.sidebar.slider(f"Bajas {eq_local} (% xG)", 0, 40, 0, 5) / 100.0
    desc_l_ui = st.sidebar.number_input(f"Descanso {eq_local} días", 1, 14, 7)

    st.sidebar.subheader(f"✈️ {eq_visit} (Elo: {int(elo_v_val)})")
    forma_v_ui = st.sidebar.slider(f"Forma {eq_visit} (%)", -25, 25, int(m_vis['forma']*100), 1) / 100.0
    bajas_v_ui = st.sidebar.slider(f"Bajas {eq_visit} (% xG)", 0, 40, 0, 5) / 100.0
    desc_v_ui = st.sidebar.number_input(f"Descanso {eq_visit} días", 1, 14, 4)

    st.sidebar.markdown("---")
    st.sidebar.header("💰 Cuotas y Gestión")
    cuota_1 = st.sidebar.number_input(f"Cuota 1", 1.01, 100.0, 2.10, 0.05)
    cuota_x = st.sidebar.number_input(f"Cuota X", 1.01, 100.0, 3.40, 0.05)
    cuota_2 = st.sidebar.number_input(f"Cuota 2", 1.01, 100.0, 3.50, 0.05)
    bankroll = st.sidebar.number_input("Bankroll (€)", 10.0, 1000000.0, 1000.0, 50.0)
    frac_kelly = st.sidebar.selectbox("Kelly", [0.25, 0.50, 1.00], index=0, format_func=lambda x: f"Kelly {int(x*100)}%")

    res = ejecutar_modelo_completo(
        m_loc, m_vis, elo_l_val, elo_v_val,
        forma_l_ui, forma_v_ui, bajas_l_ui, bajas_v_ui,
        desc_l_ui, desc_v_ui, cuota_1, cuota_x, cuota_2, campo_neutral=campo_neutral
    )

    st.markdown("---")
    
    if "🟢" in res['opinion']:
        st.success(f"🧠 **OPINIÓN DE APUESTA — {res['opinion']}**\n\n{res['detalle_opinion']}")
    elif "🟡" in res['opinion']:
        st.warning(f"🧠 **OPINIÓN DE APUESTA — {res['opinion']}**\n\n{res['detalle_opinion']}")
    else:
        st.error(f"🧠 **OPINIÓN DE APUESTA — {res['opinion']}**\n\n{res['detalle_opinion']}")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(f"1 - {eq_local}", f"{res['p1']*100:.1f}%", f"Justa: {1/res['p1']:.2f}")
    k2.metric("X - Empate", f"{res['px']*100:.1f}%", f"Justa: {1/res['px']:.2f}")
    k3.metric(f"2 - {eq_visit}", f"{res['p2']*100:.1f}%", f"Justa: {1/res['p2']:.2f}")
    k4.metric("Mas de dos goles", f"{res['mas_dos_goles']*100:.1f}%", f"Cuota: {1/res['mas_dos_goles']:.2f}")
    k5.metric("Ambos equipos marcarán", f"{res['ambos_marcaran']*100:.1f}%", f"Cuota: {1/res['ambos_marcaran']:.2f}")

    res_disc = predecir_corners_y_tarjetas(df_hist, eq_local, eq_visit)
    if res_disc:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Córners Esperados", f"{res_disc['corners_esp']:.1f}")
        c2.metric("Over 9.5 Córners", f"{res_disc['over_95_c']*100:.1f}%")
        c3.metric("Tarjetas Esperadas", f"{res_disc['tarjetas_esp']:.1f}")
        c4.metric("Over 4.5 Tarjetas", f"{res_disc['over_45_t']*100:.1f}%")

    st.markdown("### 💎 Valor Esperado (EV+) y Kelly")
    ev1, k_pct1, eur1 = evaluar_ev_kelly(res['p1'], cuota_1, frac_kelly, bankroll)
    evx, k_pctx, eurx = evaluar_ev_kelly(res['px'], cuota_x, frac_kelly, bankroll)
    ev2, k_pct2, eur2 = evaluar_ev_kelly(res['p2'], cuota_2, frac_kelly, bankroll)

    df_ev = pd.DataFrame([
        {"Apuesta": f"1 - {eq_local}", "Prob": f"{res['p1']*100:.1f}%", "Cuota Justa": round(1/res['p1'], 2), "Cuota Casa": cuota_1, "EV (%)": f"{ev1:+.2f}%", "Stake": f"{k_pct1:.2f}% ({eur1:.2f} €)", "Estado": "✅ EV+" if ev1 > 0 else "❌ No"},
        {"Apuesta": "X - Empate", "Prob": f"{res['px']*100:.1f}%", "Cuota Justa": round(1/res['px'], 2), "Cuota Casa": cuota_x, "EV (%)": f"{evx:+.2f}%", "Stake": f"{k_pctx:.2f}% ({eurx:.2f} €)", "Estado": "✅ EV+" if evx > 0 else "❌ No"},
        {"Apuesta": f"2 - {eq_visit}", "Prob": f"{res['p2']*100:.1f}%", "Cuota Justa": round(1/res['p2'], 2), "Cuota Casa": cuota_2, "EV (%)": f"{ev2:+.2f}%", "Stake": f"{k_pct2:.2f}% ({eur2:.2f} €)", "Estado": "✅ EV+" if ev2 > 0 else "❌ No"}
    ])
    st.dataframe(df_ev, use_container_width=True, hide_index=True)

with tab_jornada:
    st.subheader("📋 Pronósticos de Jornada")
    df_p = descargar_proximos_partidos()
    p_jor = df_p[df_p['liga'] == cod_liga_actual] if cod_liga_actual and not df_p.empty and 'liga' in df_p.columns else pd.DataFrame()
    if p_jor.empty:
        pares = [(equipos_lista[i], equipos_lista[i+1]) for i in range(0, min(len(equipos_lista)-1, 14), 2)]
        p_jor = pd.DataFrame(pares, columns=['local', 'visitante'])

    filas = []
    for _, row in p_jor.iterrows():
        l, v = row['local'], row['visitante']
        if l in elos_dict and v in elos_dict:
            ml = obtener_metricas_fiables(df_hist, l, elos_dict)
            mv = obtener_metricas_fiables(df_hist, v, elos_dict)
            rj = ejecutar_modelo_completo(ml, mv, elos_dict[l], elos_dict[v], ml['forma'], mv['forma'], 0.0, 0.0, 6, 6, 2.0, 3.3, 3.5, campo_neutral=campo_neutral)
            filas.append({
                "Encuentro": f"{l} vs {v}",
                "Opinión": rj['opinion'].split()[0],
                "1 (%)": f"{rj['p1']*100:.1f}%", "X (%)": f"{rj['px']*100:.1f}%", "2 (%)": f"{rj['p2']*100:.1f}%",
                "🎯 Sugerido": f"{rj['marcador_rec']} ({rj['prob_rec']*100:.1f}%)",
                "Mas de dos goles": f"{rj['mas_dos_goles']*100:.1f}%", "Ambos equipos marcarán": f"{rj['ambos_marcaran']*100:.1f}%"
            })
    if filas:
        st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)