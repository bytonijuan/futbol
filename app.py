import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Predictor Pro: Modelo Bayesiano Dixon-Coles", page_icon="⚽", layout="wide")

# ==========================================
# 1. CATÁLOGO DE COMPETICIONES MUNDIALES
# ==========================================
TORNEOS_INTERNACIONALES_CLUBES = {
    "🏆 UEFA Champions League": "UCL",
    "🟠 UEFA Europa League": "UEL",
    "🟢 UEFA Conference League": "UECL",
    "🌎 CONMEBOL Copa Libertadores / Mundial de Clubes": "LIB"
}

TORNEOS_SELECCIONES = {
    "🌍 Mundial FIFA / Clasificatorios": "WC",
    "🇪🇺 Eurocopa / UEFA Nations League": "EURO",
    "🏆 Copa América / CONMEBOL": "CA"
}

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

LIGAS_EXTRA_MUNDO = {
    "🇦🇷 Argentina - Liga Profesional": "ARG",
    "🇧🇷 Brasil - Brasileirão Série A": "BRA",
    "🇲🇽 México - Liga MX": "MEX",
    "🇺🇸 EE. UU. - MLS": "USA",
    "🇯🇵 Japón - J1 League": "JPN",
    "🇦🇹 Austria - Bundesliga": "AUT",
    "🇨🇭 Suiza - Super League": "SWZ",
    "🇩🇰 Dinamarca - Superliga": "DNK",
    "🇸🇪 Suecia - Allsvenskan": "SWE",
    "🇳🇴 Noruega - Eliteserien": "NOR"
}

# ==========================================
# 2. CONECTORES DE DATOS Y ELO DINÁMICO
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
        if codigo_liga in LIGAS_PRINCIPALES.values():
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
        else:
            url = f"https://www.football-data.co.uk/new/{codigo_liga}.csv"
            df = pd.read_csv(url, encoding='latin1')
            df = df.rename(columns={'Date': 'fecha', 'Home': 'local', 'Away': 'visitante', 'HG': 'goles_l', 'AG': 'goles_v'})

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
def cargar_pool_internacional_clubes(tipo_torneo):
    if tipo_torneo in ["UCL", "UEL", "UECL"]:
        codigos = ["SP1", "E0", "I1", "D1", "F1", "P1", "N1", "B1", "T1", "SC0"]
    else:
        codigos = ["ARG", "BRA", "MEX", "USA", "SP1", "E0", "I1", "D1"]
    dfs = []
    for c in codigos:
        df_c = descargar_liga_nacional(c)
        if not df_c.empty:
            dfs.append(df_c)
    return pd.concat(dfs, ignore_index=True).sort_values('fecha') if dfs else pd.DataFrame()

@st.cache_data(ttl=3600)
def descargar_selecciones_nacionales():
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    try:
        df = pd.read_csv(url)
        df = df.rename(columns={
            'date': 'fecha', 'home_team': 'local', 'away_team': 'visitante',
            'home_score': 'goles_l', 'away_score': 'goles_v'
        })
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        fecha_corte = df['fecha'].max() - pd.Timedelta(days=365*4)
        df = df[df['fecha'] >= fecha_corte].dropna(subset=['goles_l', 'goles_v']).copy()
        df['xg_l'] = df['goles_l']
        df['xg_v'] = df['goles_v']
        return df.sort_values('fecha')
    except Exception:
        return pd.DataFrame()

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
# 3. MOTOR ESTADÍSTICO Y CÓRNERS/TARJETAS
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

def ejecutar_modelo_completo(m_loc, m_vis, elo_l, elo_v, forma_l, forma_v, bajas_l, bajas_v, desc_l, desc_v, campo_neutral=False, rho=-0.13):
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

    dc_1x = p1 + px_emp
    dc_x2 = p2 + px_emp
    dnb_1 = p1 / (p1 + p2)
    dnb_2 = p2 / (p1 + p2)

    goles_totales = np.add.outer(range(max_g), range(max_g))
    over_15 = float(matriz[goles_totales > 1.5].sum())
    over_25 = float(matriz[goles_totales > 2.5].sum())
    over_35 = float(matriz[goles_totales > 3.5].sum())

    cs_local = float(matriz[:, 0].sum())
    cs_visit = float(matriz[0, :].sum())
    btts_no = cs_local + cs_visit - float(matriz[0, 0])
    btts_si = 1.0 - btts_no

    idx_moda = np.unravel_index(np.argmax(matriz), matriz.shape)
    marcador_moda = f"{idx_moda[0]} - {idx_moda[1]}"
    prob_moda = float(matriz[idx_moda])

    if p1 >= 0.48 and p1 > p2:
        mat_sub = np.tril(matriz, -1)
    elif p2 >= 0.48 and p2 > p1:
        mat_sub = np.triu(matriz, 1)
    elif abs(p1 - p2) < 0.12 or px_emp >= 0.30:
        mat_sub = np.diag( np.diag(matriz) )
    else:
        mat_sub = matriz
    idx_rec = np.unravel_index(np.argmax(mat_sub), mat_sub.shape)
    marcador_recomendado = f"{idx_rec[0]} - {idx_rec[1]}"
    prob_recomendado = float(matriz[idx_rec])

    entropia = - (p1 * np.log(max(1e-9, p1)) + px_emp * np.log(max(1e-9, px_emp)) + p2 * np.log(max(1e-9, p2)))
    claridad = (1.0 - (entropia / np.log(3.0))) * 100.0

    return {
        'lam_l': lam_l, 'lam_v': lam_v, 'goles_esp_tot': lam_l + lam_v,
        'p1': p1, 'px': px_emp, 'p2': p2, 'matriz': matriz,
        'dc_1x': dc_1x, 'dc_x2': dc_x2, 'dnb_1': dnb_1, 'dnb_2': dnb_2,
        'over_15': over_15, 'over_25': over_25, 'over_35': over_35,
        'btts_si': btts_si, 'btts_no': btts_no, 'cs_l': cs_local, 'cs_v': cs_visit,
        'marcador_moda': marcador_moda, 'prob_moda': prob_moda,
        'marcador_rec': marcador_recomendado, 'prob_rec': prob_recomendado,
        'claridad': claridad
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
# 4. INTERFAZ PRINCIPAL
# ==========================================
st.title("⚽ Predictor Estadístico Profesional (Bayesiano + Dixon-Coles)")

col_cat, col_comp, col_neutral = st.columns([1.1, 1.5, 0.9])

with col_cat:
    categoria = st.selectbox(
        "1️⃣ Categoría:",
        ["🏟️ Ligas Nacionales (Europa, América y Asia)",
         "🏆 Copas Internacionales de Clubes (Champions, Libertadores...)",
         "🌍 Selecciones Nacionales (Mundial, Eurocopa, Copa América)"]
    )

es_neutral_default = False
cod_liga_actual = None

if "Copas Internacionales" in categoria:
    with col_comp:
        torneo_sel = st.selectbox("2️⃣ Selecciona Competición:", list(TORNEOS_INTERNACIONALES_CLUBES.keys()))
        cod_torneo = TORNEOS_INTERNACIONALES_CLUBES[torneo_sel]
    df_hist = cargar_pool_internacional_clubes(cod_torneo)
    elos_dict = calcular_elo_con_margen(df_hist, ventaja_campo=60.0)
    df_clubelo = cargar_club_elo_vivo()
    if not df_clubelo.empty:
        mapa_clubelo = dict(zip(df_clubelo['Club'], df_clubelo['Elo']))
        for eq in elos_dict:
            if eq in mapa_clubelo:
                elos_dict[eq] = mapa_clubelo[eq]
elif "Selecciones Nacionales" in categoria:
    with col_comp:
        torneo_sel = st.selectbox("2️⃣ Selecciona Competición:", list(TORNEOS_SELECCIONES.keys()))
    df_hist = descargar_selecciones_nacionales()
    elos_dict = calcular_elo_con_margen(df_hist, ventaja_campo=40.0)
    es_neutral_default = True
else:
    TODAS_LIGAS = {**LIGAS_PRINCIPALES, **LIGAS_EXTRA_MUNDO}
    with col_comp:
        torneo_sel = st.selectbox("2️⃣ Selecciona Liga Doméstica:", list(TODAS_LIGAS.keys()))
        cod_liga_actual = TODAS_LIGAS[torneo_sel]
    df_hist = descargar_liga_nacional(cod_liga_actual)
    elos_dict = calcular_elo_con_margen(df_hist, ventaja_campo=60.0)

with col_neutral:
    st.write("")
    st.write("")
    campo_neutral = st.checkbox("🏟️ Sede Neutral", value=es_neutral_default)

if df_hist.empty:
    st.error("No se pudieron cargar los datos de esta competición.")
    st.stop()

equipos_lista = sorted(list(elos_dict.keys()))
tab_partido, tab_jornada = st.tabs(["🔍 Análisis de Partido + Kelly", "📋 Pronóstico de Jornada"])

with tab_partido:
    c_eq1, c_eq2 = st.columns(2)
    with c_eq1:
        idx1 = equipos_lista.index("Alaves") if "Alaves" in equipos_lista else 0
        eq_local = st.selectbox("Local:", equipos_lista, index=idx1)
    with c_eq2:
        idx2 = equipos_lista.index("Real Madrid") if "Real Madrid" in equipos_lista else min(1, len(equipos_lista)-1)
        eq_visit = st.selectbox("Visitante:", equipos_lista, index=idx2)

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

    res = ejecutar_modelo_completo(
        m_loc, m_vis, elo_l_val, elo_v_val,
        forma_l_ui, forma_v_ui, bajas_l_ui, bajas_v_ui,
        desc_l_ui, desc_v_ui, campo_neutral=campo_neutral
    )

    st.sidebar.markdown("---")
    st.sidebar.header("💰 Cuotas y Kelly")
    cuota_1 = st.sidebar.number_input(f"Cuota 1", 1.01, 100.0, round(max(1.05, (1/res['p1'])*0.95), 2), 0.05)
    cuota_x = st.sidebar.number_input(f"Cuota X", 1.01, 100.0, round(max(1.05, (1/res['px'])*0.95), 2), 0.05)
    cuota_2 = st.sidebar.number_input(f"Cuota 2", 1.01, 100.0, round(max(1.05, (1/res['p2'])*0.95), 2), 0.05)
    bankroll = st.sidebar.number_input("Bankroll (€)", 10.0, 1000000.0, 1000.0, 50.0)
    frac_kelly = st.sidebar.selectbox("Kelly", [0.25, 0.50, 1.00], index=0, format_func=lambda x: f"Kelly {int(x*100)}%")

    st.markdown("---")
    signo_rec = f"Victoria {eq_local} (1)" if res['p1'] > max(res['px'], res['p2']) else (f"Victoria {eq_visit} (2)" if res['p2'] > max(res['p1'], res['px']) else "Empate (X)")
    st.success(
        f"🎯 **VEREDICTO:** Tendencia principal **{signo_rec}** | "
        f"**Resultado Sugerido:** `{eq_local} {res['marcador_rec']} {eq_visit}` ({res['prob_rec']*100:.1f}%) | "
        f"**Goles Totales Esp:** `{res['goles_esp_tot']:.2f}`"
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(f"1 - {eq_local}", f"{res['p1']*100:.1f}%", f"Cuota Justa: {1/res['p1']:.2f}")
    k2.metric("X - Empate", f"{res['px']*100:.1f}%", f"Cuota Justa: {1/res['px']:.2f}")
    k3.metric(f"2 - {eq_visit}", f"{res['p2']*100:.1f}%", f"Cuota Justa: {1/res['p2']:.2f}")
    k4.metric(f"λ Local / Visit", f"{res['lam_l']:.2f} - {res['lam_v']:.2f}", f"DNB 1: {res['dnb_1']*100:.1f}%")
    k5.metric("Claridad", f"{res['claridad']:.1f} / 100")

    # Córners y Tarjetas
    res_disc = predecir_corners_y_tarjetas(df_hist, eq_local, eq_visit)
    if res_disc:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Córners Esperados", f"{res_disc['corners_esp']:.1f}")
        c2.metric("Prob. Over 9.5 Córners", f"{res_disc['over_95_c']*100:.1f}%")
        c3.metric("Tarjetas Esperadas", f"{res_disc['tarjetas_esp']:.1f}")
        c4.metric("Prob. Over 4.5 Tarjetas", f"{res_disc['over_45_t']*100:.1f}%")

    # Detector EV+
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
    st.subheader(f"📋 Pronósticos de Jornada — {torneo_sel}")
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
            rj = ejecutar_modelo_completo(ml, mv, elos_dict[l], elos_dict[v], ml['forma'], mv['forma'], 0.0, 0.0, 6, 6, campo_neutral=campo_neutral)
            filas.append({
                "Encuentro": f"{l} vs {v}",
                "1 (%)": f"{rj['p1']*100:.1f}%", "X (%)": f"{rj['px']*100:.1f}%", "2 (%)": f"{rj['p2']*100:.1f}%",
                "🎯 Resultado Sugerido": f"{rj['marcador_rec']} ({rj['prob_rec']*100:.1f}%)",
                "Over 2.5": f"{rj['over_25']*100:.1f}%", "BTTS": f"{rj['btts_si']*100:.1f}%"
            })
    if filas:
        st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)