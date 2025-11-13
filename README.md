# Análisis de Tasas de Muones por Ángulo

Este proyecto contiene el script `analysis/muonalyze.py`, el cual procesa archivos de datos por ángulo y genera:

1. Gráficas individuales Eventos/s vs Tiempo para cada ángulo.
2. Gráfico combinado Tiempo vs Tasa para todos los ángulos seleccionados.
3. Gráfico Ángulo vs Tasa Normalizada I(θ)/I(0°) con curvas de referencia y ajustes estadísticos.
4. (Opcional) Archivo CSV con estadísticas por ángulo y parámetros de ajuste.

La finalidad es estudiar la dependencia angular del flujo de muones y comparar los datos con el comportamiento teórico aproximado ∝ cos²θ o con un ajuste libre cos^nθ.

---
## Formato de datos esperado
Cada archivo se denomina `<angulo>grados` o `<angulo>grados2` según la variante. Ejemplo: `45grados`, `45grados2`.

El script parsea líneas con columnas (simplificado):
```
idx timestamp_ms adc1 adc2 sipm_mv v5 v6 deadtime_us coincident name
```
Se filtran comentarios y líneas mal formadas. Solo se usan los eventos coincidentes (`coincident == 1`).

---
## Lógica de Cálculo
### 1. Serie temporal por segundo
- Se toma el tiempo relativo acumulado (de ms a s).
- Se agrupan eventos en bins de tamaño `--bin-seconds` (por defecto 1 s).
- Se descarta el último bin parcial para evitar picos artificiales.
- Se grafica: tasa instantánea por bin, media móvil (`--rolling-seconds`), banda ±1σ local.

### 2. Tasa media por ángulo
Para cada ángulo:
- Intervalo temporal restringido por `--start-ms` y `--max-ms` (por defecto 0–600000 ms).
- Se determina el tiempo máximo real dentro de la ventana y se trunca a múltiplos completos de `bin_seconds`.
- Se calcula:
  - Número de eventos coincidentes N.
  - Duración efectiva Δt (s) tras truncado.
  - Tasa media: \( R_\theta = N/\Delta t \).
  - Error estadístico (Poisson): \( \sigma_\theta = \sqrt{N}/\Delta t \).

### 3. Normalización
- Se selecciona el ángulo de referencia 0° si está presente.
- Tasa normalizada: \( I(\theta)/I(0°) = R_\theta / R_{0} \).
- Propagación de incertidumbre (variables independientes):
$$\sigma_{\text{norm}} = \sqrt{\left(\frac{\sigma_\theta}{R_0}\right)^2 + \left(\frac{R_\theta\,\sigma_0}{R_0^2}\right)^2 }$$

### 4. Ajuste cos^nθ
- Se usan los puntos normalizados de ángulos 0°–90° (se excluye 360° si existe).
- Búsqueda en rejilla n ∈ [0,5] (paso 0.01) + refinamiento local ±0.05.
- Función: \( f(\theta) = \cos^n(\theta) \) (θ en radianes para el ajuste interno).
- Estadístico:
$$\chi^2 = \sum_i \left(\frac{y_i - f(\theta_i)}{\sigma_i}\right)^2, \quad \chi^2_{red} = \chi^2 / \text{dof}$$
- dof = N − 1 (un parámetro n).

### 5. Ajuste lineal
- Modelo: \( I(\theta) = a + b\theta \) (θ en grados o radianes según `--angle-radians`).
- Ajuste por mínimos cuadrados ponderados (pesos 1/σ²).
- dof = N − 2.

### 6. Curva teórica cos²θ
- Se dibuja si se usa `--show-cos2` para comparar visualmente con los datos.

### 7. Exportación CSV
Si se usa `--angle-stats-csv` se escribe un archivo con columnas:
- angle, counts, duration_s, mean_rate, rate_err
- normalized_rate, normalized_err
- (si ajustes activados) parámetros de cos^n y lineal con sus χ²_red y dof.

---
## Uso (Ejemplos)
Requiere Python ≥ 3.10 y paquetes: numpy, pandas, matplotlib.

### Entorno virtual (opcional)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib
```

### Ejemplo básico
```bash
python3 analysis/muonalyze.py \
  --data-dir second \
  --variant 1 \
  --out salida_basica
```

### Seleccionar ángulos explícitos
```bash
python3 analysis/muonalyze.py \
  --data-dir second \
  --variant 1 \
  --angles 0,45,90 \
  --out salida_0_45_90
```

### Rango con paso
```bash
python3 analysis/muonalyze.py \
  --data-dir second \
  --variant 1 \
  --angles 0-90:15 \
  --out salida_rango
```

### Ventana temporal acotada (0–300 s)
```bash
python3 analysis/muonalyze.py \
  --data-dir second \
  --variant 1 \
  --start-ms 0 --max-ms 300000 \
  --out salida_0_300s
```

### Gráfico normalizado con ajustes y CSV
```bash
python3 analysis/muonalyze.py \
  --data-dir second \
  --variant 1 \
  --angles 0,45,90 \
  --start-ms 0 --max-ms 300000 \
  --fit-cosn --fit-linear --show-cos2 --angle-radians \
  --angle-stats-csv salida_0_45_90/angle_stats.csv \
  --out salida_0_45_90
```

### Solo estadísticas sin gráfico angular
```bash
python3 analysis/muonalyze.py \
  --data-dir second \
  --variant 1 \
  --angles 0,45,90 \
  --angle-stats-csv solo_stats/angle_stats.csv \
  --skip-angle-plot \
  --out solo_stats
```

---
## Interpretación de Parámetros
- **mean_rate** y **rate_err**: Flujo promedio y su error estadístico (Poisson) durante Δt.
- **normalized_rate**: Flujo relativo a 0°. Útil para comparar forma angular.
- **n (cos^n)**: Exponente que aplana o curva la función; valores típicos teóricos para muones cercanos a vertical ~2.
- **a, b (lineal)**: Parámetros de una aproximación simplificada; b indica pendiente (decaimiento por radian/grado). Con pocos puntos puede sobre-ajustar.
- **χ², χ²_red, dof**: Calidad del ajuste. χ²_red ≈ 1 sugiere consistencia entre modelo y errores. Valores muy bajos pueden indicar sobre-ajuste, muy altos mal modelo o errores subestimados.

---
## Limitaciones y Recomendaciones
1. Tres ángulos (0°,45°,90°) no bastan para distinguir bien cos^n de una recta. Usa más ángulos (15°,30°,60°,75°).
2. El ángulo 90° suele tener flujo muy reducido teóricamente; verificar alineación real si aparece alto.
3. Considerar correcciones geométricas (proyección de área ∝ cos θ) antes de comparar con cos^n.
4. Aumentar tiempo de adquisición para reducir errores relativos (√N).
5. Añadir intervalo de confianza para n (región donde χ² = χ²_min + 1) si se requiere rigor.

---
## Posibles Extensiones (Next Steps)
- Panel de residuales (Datos − Modelo) para cada ajuste.
- Cálculo de AIC/BIC para comparar modelos cos^n vs lineal.
- Bootstrap para estimar incertidumbres robustas de la tasa normalizada.
- Corrección por eficiencia / dead time más avanzada.
- Análisis multi-run combinando varias carpetas de datos.

---
## Mantenimiento Rápido
- Código principal: `analysis/muonalyze.py`
- Ajustar listas de ángulos permitidos en `ALLOWED_ANGLES` si se agregan nuevos.
- Revisar parseo si cambia el formato de los archivos.

---
## Contacto / Créditos
Proyecto educativo de análisis de flujo de muones. Ajustes y fórmulas basadas en estadística básica (Poisson) y comparación simple de modelos.

