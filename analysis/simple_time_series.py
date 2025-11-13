#!/usr/bin/env python3
"""Genera gráficas Eventos/s vs Tiempo para cada ángulo usando solo una variante (1 o 2).

Uso:
    python simple_time_series.py --data-dir data --variant 1 --out output_simple --start-ms 0 --max-ms 600000

Produce un PNG por ángulo con:
  - Counts (1 s)
  - Rolling mean (3 s) (configurable)
  - Banda ±1 σ (desviación estándar local en ventana)
Además genera un gráfico combinado con todos los ángulos: Tiempo vs Tasa de Eventos con barras de error σ = √N/Δt.
Notas:
    - El tiempo de los datos está en milisegundos [ms]. Se convierte a segundos para graficar.
    - Se puede limitar el rango temporal con --start-ms y --max-ms (por defecto 0–600000 ms = 600 s).
"""
import argparse
import os
import re
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ANGLE_FILE_BASE_RE = re.compile(r"^(?P<angle>\d+)grados(2)?$")
ALLOWED_ANGLES = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,360]

# ---------------- Parsing -----------------

def parse_line(line: str):
    parts = re.split(r"\s+", line.strip())
    if len(parts) < 10:
        raise ValueError("Not enough columns")
    idx = int(parts[0])
    timestamp_ms = int(parts[1])
    # adc1, adc2, sipm no usados para esta gráfica pero parseamos igual
    adc1 = int(parts[2]); adc2 = int(parts[3]); sipm_mv = float(parts[4])
    v5 = float(parts[5]); v6 = float(parts[6])
    # Detectar temp/press por magnitud
    if v5 > 1000 and v6 < 200:
        press_pa = v5; temp_c = v6
    elif v6 > 1000 and v5 < 200:
        press_pa = v6; temp_c = v5
    else:
        temp_c = v5; press_pa = v6
    deadtime_us = int(float(parts[7]))
    coincident = int(parts[8])
    name = parts[9]
    return idx, timestamp_ms, adc1, adc2, sipm_mv, temp_c, press_pa, deadtime_us, coincident, name


def load_file(path: str) -> pd.DataFrame:
    rows = []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            if line.startswith('###'):
                continue
            try:
                rows.append(parse_line(line))
            except Exception:
                continue
    df = pd.DataFrame(rows, columns=["idx","timestamp_ms","adc1","adc2","sipm_mv","temp_c","press_pa","deadtime_us","coincident","name"])
    if not df.empty:
        t_vals = df["timestamp_ms"].to_numpy(dtype=np.int64)
        dt_ms = np.diff(t_vals, prepend=t_vals[0]); dt_ms[0] = 0
        dt_pos_ms = np.where(dt_ms > 0, dt_ms, 0)
        df["dt_pos_s"] = dt_pos_ms / 1000.0
        df["elapsed_s"] = df["dt_pos_s"].cumsum()
    return df

# --------------- Discovery ----------------

def list_angles(data_dir: str) -> List[int]:
    angles = set()
    for name in os.listdir(data_dir):
        m = ANGLE_FILE_BASE_RE.match(name)
        if m:
            angles.add(int(m.group('angle')))
    return sorted(angles)


def parse_angles_arg(arg: str) -> List[int]:
    """Parsea la cadena de --angles.
    Soporta:
      - lista separada por comas: "0,5,15,90,360"
      - rangos con paso: "0-90:5" (inclusive en ambos extremos)
      - la palabra "all" para usar todos los permitidos (ALLOWED_ANGLES)
    Filtra cualquier ángulo no permitido contra ALLOWED_ANGLES y elimina duplicados conservando el orden.
    """
    if not arg or not arg.strip():
        return []
    arg = arg.strip().lower()
    if arg in ("all", "todos", "*"):
        return list(ALLOWED_ANGLES)
    requested: List[int] = []
    allowed_set = set(ALLOWED_ANGLES)
    for token in (t.strip() for t in arg.split(',') if t.strip()):
        if '-' in token:
            # rango a-b[:step]
            rng, *step_part = token.split(':', 1)
            a_str, b_str = rng.split('-', 1)
            try:
                a = int(a_str); b = int(b_str)
            except ValueError:
                continue
            step = 1
            if step_part:
                try:
                    step = int(step_part[0])
                except ValueError:
                    step = 1
            if step <= 0:
                step = 1
            if a <= b:
                seq = list(range(a, b+1, step))
            else:
                seq = list(range(a, b-1, -step))
            for v in seq:
                if v in allowed_set and v not in requested:
                    requested.append(v)
        else:
            try:
                v = int(token)
            except ValueError:
                continue
            if v in allowed_set and v not in requested:
                requested.append(v)
    return requested


def build_filename(angle: int, variant: int) -> str:
    suffix = '2' if variant == 2 else ''
    return f"{angle}grados{suffix}"

# --------------- Plotting -----------------

def plot_per_second(df: pd.DataFrame, angle: int, outdir: str, bin_seconds: float, rolling_seconds: float, start_ms: int, max_ms: int):
    coinc = df[df['coincident'] == 1]
    if coinc.empty:
        return
    # Limitar al intervalo [start_ms, max_ms]
    max_seconds = (max_ms or 600000) / 1000.0
    min_seconds = (start_ms or 0) / 1000.0
    elapsed_all = coinc['elapsed_s'].to_numpy()
    mask_time = (elapsed_all >= min_seconds) & (elapsed_all <= max_seconds)
    elapsed = elapsed_all[mask_time]
    if len(elapsed) == 0:
        return
    T = float(min(elapsed[-1], max_seconds))
    # Para evitar picos artificiales en el último bin parcial (Δt << 1 s),
    # ignoramos el resto fraccional y usamos solo bins completos de tamaño bin_seconds.
    t_end = np.floor((T - min_seconds) / bin_seconds) * bin_seconds + min_seconds
    if t_end <= min_seconds:
        return  # no hay al menos un bin completo
    edges = np.arange(min_seconds, t_end + bin_seconds, bin_seconds)
    # Refiltrar por seguridad al rango [min_seconds, t_end]
    elapsed = elapsed[(elapsed >= min_seconds) & (elapsed <= t_end)]
    counts, edges = np.histogram(elapsed, bins=edges)
    widths = np.diff(edges)
    centers = edges[:-1] + widths/2.0
    with np.errstate(divide='ignore', invalid='ignore'):
        rates = np.where(widths > 0, counts / widths, np.nan)
    window_bins = max(int(rolling_seconds / bin_seconds), 1)
    rm = pd.Series(rates).rolling(window=window_bins, center=True, min_periods=1).mean().to_numpy()
    rs = pd.Series(rates).rolling(window=window_bins, center=True, min_periods=1).std(ddof=0).fillna(0).to_numpy()
    mu = np.nanmean(rates)
    sigma_global = np.nanstd(rates, ddof=0)

    plt.figure(figsize=(8,3.2))
    plt.plot(centers, rates, linewidth=0.7, alpha=0.65, label=f"Counts ({bin_seconds:g} s)")
    plt.plot(centers, rm, color='orange', linewidth=1.2, label=f"Rolling mean ({rolling_seconds:.0f} s)")
    plt.fill_between(centers, rm - rs, rm + rs, color='skyblue', alpha=0.3, label='±1 σ')
    plt.xlabel('Time [s]'); plt.ylabel('Events / s'); plt.title(f'Ángulo {angle}°')
    plt.legend(fontsize=8)
    txt = f"N = {len(elapsed)}\nμ = {mu:.2f} 1/s\nσ = {sigma_global:.2f} 1/s"
    plt.text(0.98, 0.02, txt, transform=plt.gca().transAxes, va='bottom', ha='right', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"per_second_angle_{angle}.png")
    plt.savefig(outpath, dpi=140)
    plt.close()

def plot_time_vs_rate_all(angle_data: Dict[int, pd.DataFrame], outdir: str, bin_seconds: float, start_ms: int, max_ms: int):
    """Grafica todos los ángulos en un mismo gráfico: Tiempo vs Tasa de eventos.
    - Usa solo eventos coincidentes, limitados a los primeros max_ms.
    - Barras de error estadísticas: σ = √N / Δt por bin.
    - Evita el último bin vacío usando el ancho real del último intervalo.
    """
    if not angle_data:
        return
    plt.figure(figsize=(8,5))
    max_seconds = (max_ms or 600000) / 1000.0
    min_seconds = (start_ms or 0) / 1000.0
    for angle in sorted(angle_data.keys()):
        df = angle_data[angle]
        coinc = df[df['coincident'] == 1]
        if coinc.empty:
            continue
        elapsed_all = coinc['elapsed_s'].to_numpy()
        elapsed = elapsed_all[(elapsed_all >= min_seconds) & (elapsed_all <= max_seconds)]
        if len(elapsed) == 0:
            continue
        # Tiempo máximo real limitado por ventana
        T = float(min(elapsed[-1], max_seconds))
        # Descartar bin parcial final: usar solo múltiplos completos de bin_seconds
        t_end = np.floor((T - min_seconds) / bin_seconds) * bin_seconds + min_seconds
        if t_end <= min_seconds:
            continue  # no hay al menos un bin completo
        edges = np.arange(min_seconds, t_end + bin_seconds, bin_seconds)
        elapsed = elapsed[(elapsed >= min_seconds) & (elapsed <= t_end)]
        hist, edges = np.histogram(elapsed, bins=edges)
        widths = np.diff(edges)
        centers = edges[:-1] + widths/2.0
        with np.errstate(divide='ignore', invalid='ignore'):
            rates = np.where(widths > 0, hist / widths, np.nan)
            errs = np.where(widths > 0, np.sqrt(hist) / widths, np.nan)
        plt.errorbar(centers, rates, yerr=errs, fmt='-o', ms=3, capsize=2, alpha=0.85, label=f'{angle}°')
    plt.xlabel('Time [s]')
    plt.ylabel('Events / s')
    plt.title(f'Tiempo vs Tasa de eventos (todos los ángulos) | intervalo [{min_seconds:.0f}, {max_seconds:.0f}] s')
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, 'tiempo_vs_tasa.png'), dpi=150)
    plt.close()

# --------------- Angle normalized rate & cos^n fit -----------------

def compute_mean_rate(angle_data: Dict[int, pd.DataFrame], bin_seconds: float, start_ms: int, max_ms: int) -> pd.DataFrame:
    """Calcula la tasa media por ángulo en el intervalo solicitado usando solo eventos coincidentes.
    La tasa se define como total_eventos / duración, donde la duración se trunca a múltiplos
    completos de bin_seconds para evitar sesgo de bin parcial.
    Devuelve DataFrame con columnas: angle, counts, duration_s, mean_rate, rate_err.
    """
    records = []
    min_seconds = (start_ms or 0) / 1000.0
    max_seconds_window = (max_ms or 600000) / 1000.0
    for ang, df in angle_data.items():
        coinc = df[df['coincident'] == 1]
        if coinc.empty:
            continue
        elapsed_all = coinc['elapsed_s'].to_numpy()
        elapsed = elapsed_all[(elapsed_all >= min_seconds) & (elapsed_all <= max_seconds_window)]
        if len(elapsed) == 0:
            continue
        T = float(min(elapsed[-1], max_seconds_window))
        t_end = np.floor((T - min_seconds) / bin_seconds) * bin_seconds + min_seconds
        if t_end <= min_seconds:
            continue
        # Filtrar definitivo
        elapsed = elapsed[(elapsed >= min_seconds) & (elapsed <= t_end)]
        counts = len(elapsed)
        duration = t_end - min_seconds
        if duration <= 0:
            continue
        mean_rate = counts / duration
        rate_err = np.sqrt(counts) / duration  # Propagación Poisson
        records.append({
            'angle': ang,
            'counts': counts,
            'duration_s': duration,
            'mean_rate': mean_rate,
            'rate_err': rate_err
        })
    df_rates = pd.DataFrame(records).sort_values('angle')
    return df_rates

def normalize_rates(df_rates: pd.DataFrame, ref_angle: int = 0) -> pd.DataFrame:
    """Normaliza mean_rate respecto al ángulo ref_angle y propaga incertidumbre.
    Añade columnas normalized_rate y normalized_err.
    """
    if df_rates.empty:
        return df_rates
    if ref_angle not in df_rates['angle'].values:
        print(f"[WARN] Ángulo de referencia {ref_angle}° no presente. No se normaliza.")
        df_rates['normalized_rate'] = np.nan
        df_rates['normalized_err'] = np.nan
        return df_rates
    ref_row = df_rates[df_rates['angle'] == ref_angle].iloc[0]
    R0 = ref_row['mean_rate']; dR0 = ref_row['rate_err']
    norm_vals = []
    norm_errs = []
    for _, row in df_rates.iterrows():
        R = row['mean_rate']; dR = row['rate_err']
        if R0 <= 0:
            norm_vals.append(np.nan); norm_errs.append(np.nan); continue
        nR = R / R0
        # Propagación: f=R/R0 => df^2 = (dR/R0)^2 + (R*dR0/R0^2)^2
        nErr = np.sqrt((dR / R0)**2 + (R * dR0 / (R0**2))**2)
        norm_vals.append(nR)
        norm_errs.append(nErr)
    df_rates['normalized_rate'] = norm_vals
    df_rates['normalized_err'] = norm_errs
    return df_rates

def fit_cosn(df_norm: pd.DataFrame, exclude_angles: Optional[List[int]] = None) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[float]]:
    """Ajusta normalized_rate ≈ cos(theta)^n mediante búsqueda de rejilla.
    Devuelve (n_best, chi2_red). Excluye ángulos en exclude_angles.
    Si no hay suficientes puntos, retorna (None, None).
    """
    if df_norm.empty:
        return None, None, None, None
    data = df_norm.copy()
    if exclude_angles:
        data = data[~data['angle'].isin(exclude_angles)]
    data = data[(data['normalized_rate'].notna()) & (data['angle'] >= 0) & (data['angle'] <= 90)]  # limitar a 0–90° típicamente
    if len(data) < 3:
        return None, None, None, None
    angles_rad = np.deg2rad(data['angle'].to_numpy(dtype=float))
    y = data['normalized_rate'].to_numpy(dtype=float)
    dy = data['normalized_err'].replace(0, np.nan).to_numpy(dtype=float)
    dy = np.where(np.isnan(dy) | (dy<=0), 1.0, dy)  # evitar división por cero
    n_grid = np.linspace(0.0, 5.0, 501)  # paso 0.01
    chi2_vals = []
    cos_vals = np.cos(angles_rad)
    for n in n_grid:
        model = cos_vals**n
        chi2 = np.sum(((y - model)/dy)**2)
        chi2_vals.append(chi2)
    chi2_arr = np.array(chi2_vals)
    idx_best = int(np.argmin(chi2_arr))
    n_best_coarse = n_grid[idx_best]
    # Refinamiento local usando pequeña ventana alrededor
    n_min = max(0.0, n_best_coarse - 0.05)
    n_max = min(5.0, n_best_coarse + 0.05)
    n_fine = np.linspace(n_min, n_max, 201)
    chi2_fine = []
    for n in n_fine:
        model = cos_vals**n
        chi2_fine.append(np.sum(((y - model)/dy)**2))
    chi2_fine = np.array(chi2_fine)
    idx_fine = int(np.argmin(chi2_fine))
    n_best = float(n_fine[idx_fine])
    dof = max(len(y) - 1, 1)
    chi2_red = chi2_fine[idx_fine] / dof
    chi2_best = float(chi2_fine[idx_fine])
    return n_best, chi2_red, int(dof), chi2_best

def fit_linear(df_norm: pd.DataFrame, use_radians: bool=False, exclude_angles: Optional[List[int]] = None) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[int], Optional[float]]:
    """Ajuste lineal I_norm ≈ a + b*theta (theta en grados o radianes según use_radians).
    Devuelve (a, b, chi2_red). Usa pesos 1/σ^2.
    Excluye ángulos en exclude_angles si se proporcionan.
    Requiere al menos 2 puntos.
    """
    if df_norm.empty:
        return None, None, None, None, None
    data = df_norm.copy()
    if exclude_angles:
        data = data[~data['angle'].isin(exclude_angles)]
    data = data[data['normalized_rate'].notna()]
    if len(data) < 2:
        return None, None, None, None, None
    theta_deg = data['angle'].to_numpy(dtype=float)
    theta = np.deg2rad(theta_deg) if use_radians else theta_deg
    y = data['normalized_rate'].to_numpy(dtype=float)
    dy = data['normalized_err'].replace(0, np.nan).to_numpy(dtype=float)
    dy = np.where(np.isnan(dy) | (dy<=0), 1.0, dy)
    w = 1.0/(dy**2)
    # Weighted least squares formulas
    W = np.sum(w)
    Wx = np.sum(w*theta)
    Wy = np.sum(w*y)
    Wxx = np.sum(w*theta*theta)
    Wxy = np.sum(w*theta*y)
    denom = W*Wxx - Wx*Wx
    if denom == 0:
        return None, None, None, None, None
    b = (W*Wxy - Wx*Wy)/denom
    a = (Wxx*Wy - Wx*Wxy)/denom
    model = a + b*theta
    chi2 = np.sum(((y - model)/dy)**2)
    dof = max(len(y)-2, 1)
    chi2_red = chi2/dof
    chi2_best = float(chi2)
    return a, b, chi2_red, int(dof), chi2_best

def plot_angle_vs_normalized(
    df_norm: pd.DataFrame,
    outdir: str,
    n_fit: Optional[float]=None,
    use_radians: bool=False,
    show_cos2: bool=False,
    linear_params: Optional[Tuple[float,float]]=None,
    cosn_stats: Optional[Tuple[float,float,int,float]]=None,
    linear_stats: Optional[Tuple[float,float,int,float]]=None,
):
    if df_norm.empty:
        print('[WARN] No hay datos para graficar ángulo vs tasa normalizada.')
        return
    plt.figure(figsize=(7.2,4.2))
    if use_radians:
        ang_vals_deg = df_norm['angle'].to_numpy(dtype=float)
        ang_vals_rad = np.deg2rad(ang_vals_deg)
        plt.errorbar(ang_vals_rad, df_norm['normalized_rate'], yerr=df_norm['normalized_err'], fmt='o', capsize=3, label='Datos normalizados')
        if show_cos2:
            ang_line_deg = np.linspace(0,90,361)
            ang_line_rad = np.deg2rad(ang_line_deg)
            plt.plot(ang_line_rad, np.cos(ang_line_rad)**2, '--', color='tab:blue', label='cos²θ')
        if n_fit is not None:
            ang_plot_deg = np.linspace(0,90,181)
            ang_plot_rad = np.deg2rad(ang_plot_deg)
            model = np.cos(ang_plot_rad)**n_fit
            plt.plot(ang_plot_rad, model, '-', label=f'Ajuste cos^n, n={n_fit:.2f}')
        if linear_params is not None:
            a_lin, b_lin = linear_params
            ang_line = np.linspace(0, np.deg2rad(90), 181)
            plt.plot(ang_line, a_lin + b_lin*ang_line, ':', color='tab:green', label=f'Ajuste lineal')
        plt.xlabel('Ángulo [rad]')
    else:
        plt.errorbar(df_norm['angle'], df_norm['normalized_rate'], yerr=df_norm['normalized_err'], fmt='o', capsize=3, label='Datos normalizados')
        if show_cos2:
            ang_line = np.linspace(0,90,361)
            plt.plot(ang_line, np.cos(np.deg2rad(ang_line))**2, '--', color='tab:blue', label='cos²θ')
        if n_fit is not None:
            ang_plot = np.linspace(0,90,181)
            model = np.cos(np.deg2rad(ang_plot))**n_fit
            plt.plot(ang_plot, model, '-', label=f'Ajuste cos^n, n={n_fit:.2f}')
        if linear_params is not None:
            a_lin, b_lin = linear_params
            ang_line = np.linspace(0,90,181)
            plt.plot(ang_line, a_lin + b_lin*ang_line, ':', color='tab:green', label='Ajuste lineal')
        plt.xlabel('Ángulo [°]')
    plt.ylabel('Tasa normalizada I(θ)/I(0°)')
    plt.title('Ángulo vs Tasa Normalizada')
    plt.grid(alpha=0.3)
    # Leyenda siempre en esquina inferior izquierda
    plt.legend(fontsize=8, loc='lower left')
    # Anotación fija: esquina inferior izquierda, encima de la leyenda
    lines = []
    if show_cos2:
        lines.append('Modelo teórico: cos²θ')
    if cosn_stats and cosn_stats[0] is not None:
        n_val, chi2_red_val, dof_cosn, chi2_val = cosn_stats
        lines.append(f'Ajuste cos^n: n={n_val:.3f}\nχ²_red={chi2_red_val:.2f} (χ²={chi2_val:.2f}, dof={dof_cosn})')
    if linear_stats and linear_stats[0] is not None:
        a_val, b_val, dof_lin, chi2_lin_val = linear_stats
        unit = 'rad' if use_radians else '°'
        lines.append(f'Lineal: I=a+bθ\na={a_val:.3f}, b={b_val:.3f} {unit}⁻¹\nχ²_red={(chi2_lin_val/max(dof_lin,1)):.2f} (χ²={chi2_lin_val:.2f}, dof={dof_lin})')
    if lines:
        txt = '\n'.join(lines)
        # Colocar en coordenadas relativas un poco arriba de la leyenda (y ~0.30 suponiendo leyenda ocupa ~0.15)
        plt.text(0.02, 0.50, txt, transform=plt.gca().transAxes, va='top', ha='left', fontsize=8,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, 'angulo_vs_tasa_normalizada.png'), dpi=160)
    plt.close()

def export_angle_stats(df_norm: pd.DataFrame, path: str, n_fit: Optional[float], chi2_red: Optional[float]):
    if df_norm.empty:
        return
    df_out = df_norm.copy()
    df_out['n_fit'] = n_fit
    df_out['chi2_red'] = chi2_red
    try:
        df_out.to_csv(path, index=False)
        print(f'[INFO] Estadísticas por ángulo exportadas a {path}')
    except Exception as e:
        print(f'[WARN] No se pudo escribir CSV {path}: {e}')

# --------------- Main ---------------------

def main():
    ap = argparse.ArgumentParser(description='Genera gráficas Eventos/s vs Tiempo por ángulo (una sola variante).')
    ap.add_argument('--data-dir', default='data', help='Directorio con archivos <angulo>grados y <angulo>grados2')
    ap.add_argument('--variant', type=int, default=1, choices=[1,2], help='Selecciona variante 1 o 2 para todas las gráficas.')
    ap.add_argument('--out', default='output_simple', help='Directorio de salida para PNGs.')
    ap.add_argument('--bin-seconds', type=float, default=1.0, help='Tamaño de bin para conteo por segundo.')
    ap.add_argument('--rolling-seconds', type=float, default=3.0, help='Ventana para media móvil.')
    ap.add_argument('--angles', type=str, default='', help='Ángulos a graficar: lista ("0,5,15,90,360"), rango con paso ("0-90:5"), o "all". Si se deja vacío, usa los archivos presentes.')
    ap.add_argument('--start-ms', type=int, default=0, help='Usar datos desde este tiempo (ms). Default 0.')
    ap.add_argument('--max-ms', type=int, default=600000, help='Usar datos solo hasta este tiempo (ms). Default 600000 (600 s).')
    ap.add_argument('--time-bin', type=float, default=60.0, help='Bin (s) para el gráfico combinado Tiempo vs Tasa.')
    ap.add_argument('--angle-stats-csv', type=str, default='', help='Ruta CSV para exportar estadísticas de tasas por ángulo (normalizadas).')
    ap.add_argument('--fit-cosn', action='store_true', help='Realiza ajuste cos^n sobre tasas normalizadas (0°–90°).')
    ap.add_argument('--fit-linear', action='store_true', help='Realiza ajuste lineal I = a + b*θ para comparar con cos^n y cos².')
    ap.add_argument('--skip-angle-plot', action='store_true', help='Si se especifica, no genera el gráfico ángulo vs tasa normalizada.')
    ap.add_argument('--angle-radians', action='store_true', help='Usar eje angular en radianes en el gráfico ángulo vs tasa normalizada.')
    ap.add_argument('--show-cos2', action='store_true', help='Dibujar curva teórica cos²θ en el gráfico ángulo vs tasa normalizada.')
    args = ap.parse_args()

    # Si el usuario especifica --angles, respetamos exactamente su lista (aunque falten archivos).
    # Si no, detectamos los ángulos presentes en la carpeta de datos.
    if args.angles.strip():
        angles = parse_angles_arg(args.angles)
        if not angles:
            print('[WARN] --angles no produjo ángulos válidos; usando detección automática de la carpeta.')
            angles = list_angles(args.data_dir)
    else:
        angles = list_angles(args.data_dir)

    if not angles:
        print('No se encontraron ángulos en el directorio.')
        return 1

    print(f"Ángulos detectados: {angles}. Variante seleccionada: {args.variant}")
    angle_data: Dict[int, pd.DataFrame] = {}
    for ang in angles:
        fname = build_filename(ang, args.variant)
        path = os.path.join(args.data_dir, fname)
        if not os.path.isfile(path):
            print(f"[WARN] Archivo no encontrado para ángulo {ang}: {fname}")
            continue
        df = load_file(path)
        if df.empty:
            print(f"[WARN] Archivo vacío o no parseable: {fname}")
            continue
        angle_data[ang] = df
        plot_per_second(df, ang, args.out, bin_seconds=args.bin_seconds, rolling_seconds=args.rolling_seconds,
                        start_ms=args.start_ms, max_ms=args.max_ms)
        print(f"Generado per_second_angle_{ang}.png")
    plot_time_vs_rate_all(angle_data, args.out, bin_seconds=args.time_bin, start_ms=args.start_ms, max_ms=args.max_ms)

    # Nuevo bloque: estadísticas por ángulo y normalización
    df_rates = compute_mean_rate(angle_data, bin_seconds=args.bin_seconds, start_ms=args.start_ms, max_ms=args.max_ms)
    df_norm = normalize_rates(df_rates, ref_angle=0)
    n_fit = None; chi2_red = None; dof_cosn = None; chi2_raw_cosn = None
    a_lin = None; b_lin = None; chi2_lin_red = None; dof_lin = None; chi2_raw_lin = None
    if args.fit_cosn:
        n_fit, chi2_red, dof_cosn, chi2_raw_cosn = fit_cosn(df_norm, exclude_angles=[360])
        if n_fit is not None:
            print(f"[INFO] Mejor n ≈ {n_fit:.3f} (χ²_red={chi2_red:.2f}, dof={dof_cosn})")
        else:
            print('[WARN] No se pudo ajustar n (datos insuficientes).')
    if args.fit_linear:
        a_lin, b_lin, chi2_lin_red, dof_lin, chi2_raw_lin = fit_linear(df_norm, use_radians=args.angle_radians, exclude_angles=[360])
        if a_lin is not None:
            print(f"[INFO] Ajuste lineal: a={a_lin:.3f}, b={b_lin:.3f} (χ²_red={chi2_lin_red:.2f}, dof={dof_lin})")
        else:
            print('[WARN] No se pudo realizar ajuste lineal (datos insuficientes).')
    if not args.skip_angle_plot:
        linear_params = (a_lin, b_lin) if a_lin is not None else None
        cosn_stats = (n_fit, chi2_red, dof_cosn, chi2_raw_cosn) if n_fit is not None else None
        linear_stats = (a_lin, b_lin, dof_lin, chi2_raw_lin) if a_lin is not None else None
        plot_angle_vs_normalized(
            df_norm,
            args.out,
            n_fit=n_fit,
            use_radians=args.angle_radians,
            show_cos2=args.show_cos2,
            linear_params=linear_params,
            cosn_stats=cosn_stats,
            linear_stats=linear_stats,
        )
        print('Generado angulo_vs_tasa_normalizada.png')
    if args.angle_stats_csv:
        # Añadir parámetros lineales al DataFrame antes de exportar
        if a_lin is not None:
            df_norm['linear_a'] = a_lin
            df_norm['linear_b'] = b_lin
            df_norm['linear_chi2_red'] = chi2_lin_red
            df_norm['linear_dof'] = dof_lin
        if n_fit is not None:
            df_norm['cosn_n'] = n_fit
            df_norm['cosn_chi2_red'] = chi2_red
            df_norm['cosn_dof'] = dof_cosn
        export_angle_stats(df_norm, args.angle_stats_csv, n_fit, chi2_red)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
