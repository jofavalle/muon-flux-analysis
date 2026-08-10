"""Tests de tasas y de binning, incluido un caso calculado a mano."""

from pathlib import Path

import numpy as np
import pytest

from lib_muon import binning, io, rates

RAIZ = Path(__file__).resolve().parents[1]
CRUDOS = RAIZ / "data" / "raw"


def corrida_sintetica(timestamps_ms, deadtime_us, coincidente=None):
    """Corrida minima para comprobar aritmetica con la respuesta a mano."""
    n = len(timestamps_ms)
    return io.Corrida(
        angulo_deg=0,
        ruta="<sintetica>",
        evento=np.arange(1, n + 1, dtype=np.int64),
        timestamp_ms=np.asarray(timestamps_ms, dtype=float),
        adc1=np.zeros(n, dtype=np.int64),
        adc2=np.zeros(n, dtype=np.int64),
        sipm_mv=np.zeros(n),
        presion_pa=np.full(n, 93600.0),
        temperatura_c=np.full(n, 28.0),
        deadtime_us=np.asarray(deadtime_us, dtype=float),
        coincidente=(np.ones(n, dtype=np.int64) if coincidente is None
                     else np.asarray(coincidente, dtype=np.int64)),
        arranque_sensor=np.zeros(n, dtype=bool),
        calidad=io.InformeCalidad(ruta="<sintetica>"),
    )


class TestCasoCalculadoAMano:
    """11 eventos a un evento por segundo, con 1 s de tiempo muerto total."""

    def test_aritmetica_completa(self):
        corrida = corrida_sintetica(
            timestamps_ms=np.arange(11) * 1000.0,
            deadtime_us=np.full(11, 100_000.0 / 11),  # 100 ms repartidos: 0.1 s
        )
        tasa = rates.calcular_tasa(corrida)

        # A mano: ventana = t11 - t1 = 10000 ms = 10 s exactos.
        assert tasa.ventana_s == pytest.approx(10.0)
        # 11 eventos, 10 intervalos.
        assert tasa.n_eventos == 11
        assert tasa.n_intervalos == 10
        # Tiempo muerto = 100000 us = 0.1 s. Vivo = 10 - 0.1 = 9.9 s.
        assert tasa.tiempo_muerto_s == pytest.approx(0.1)
        assert tasa.tiempo_vivo_s == pytest.approx(9.9)
        # Tasa de pared = 10/10 = 1 Hz exacto.
        assert tasa.tasa_pared == pytest.approx(1.0)
        # Tasa viva = 10/9.9 = 1.0101... Hz.
        assert tasa.tasa_viva == pytest.approx(10.0 / 9.9)
        # Error relativo = 1/sqrt(10).
        assert tasa.error_tasa_viva == pytest.approx(tasa.tasa_viva / np.sqrt(10))

    def test_el_estimador_usa_n_menos_uno(self):
        """Para N eventos en [t1, tN] hay N-1 intervalos, no N. El script
        original dividia por N y sobrestimaba la tasa en 1/N."""
        corrida = corrida_sintetica(
            timestamps_ms=np.arange(101) * 1000.0, deadtime_us=np.zeros(101)
        )
        tasa = rates.calcular_tasa(corrida)
        assert tasa.tasa_pared == pytest.approx(1.0)      # 100/100, correcto
        sesgado = 101 / 100.0                              # lo que hacia el original
        assert sesgado == pytest.approx(1.01)
        assert tasa.tasa_pared != pytest.approx(sesgado)

    def test_tiempo_vivo_sube_la_tasa(self):
        corrida = corrida_sintetica(
            timestamps_ms=np.arange(11) * 1000.0, deadtime_us=np.full(11, 1e6 / 11)
        )
        tasa = rates.calcular_tasa(corrida)
        assert tasa.tiempo_vivo_s == pytest.approx(9.0)
        assert tasa.tasa_viva > tasa.tasa_pared
        assert tasa.tasa_viva == pytest.approx(10.0 / 9.0)

    def test_pocos_eventos_lanza(self):
        corrida = corrida_sintetica([0.0], [0.0])
        with pytest.raises(ValueError, match="al menos 2"):
            rates.calcular_tasa(corrida)


class TestNormalizacion:
    def test_el_angulo_de_referencia_da_exactamente_uno(self):
        corridas = io.load_all(CRUDOS)
        tasas = {a: rates.calcular_tasa(c) for a, c in corridas.items()}
        normalizadas = rates.normalizar(tasas, angulo_ref=0)
        assert normalizadas[0][0] == pytest.approx(1.0)

    def test_falta_la_referencia_lanza(self):
        corridas = io.load_all(CRUDOS)
        tasas = {a: rates.calcular_tasa(c) for a, c in corridas.items()}
        with pytest.raises(ValueError, match="referencia"):
            rates.normalizar(tasas, angulo_ref=77)


class TestBinning:
    def test_descarta_el_bin_parcial_final(self):
        """10.5 s de datos con bins de 1 s dan 10 bins, no 11."""
        tiempos = np.arange(0, 10.5, 0.1)
        centros, cuentas, anchos = binning.bins_completos(tiempos, bin_s=1.0,
                                                          inicio_s=0.0, fin_s=10.5)
        assert len(centros) == 10
        assert np.all(anchos == 1.0)

    def test_cuentas_correctas_en_un_caso_trivial(self):
        tiempos = np.array([0.5, 1.5, 1.6, 2.5])
        centros, cuentas, _ = binning.bins_completos(tiempos, bin_s=1.0,
                                                     inicio_s=0.0, fin_s=3.0)
        np.testing.assert_array_equal(cuentas, [1.0, 2.0, 1.0])
        np.testing.assert_allclose(centros, [0.5, 1.5, 2.5])

    def test_bin_mayor_que_los_datos_no_da_bins(self):
        centros, cuentas, _ = binning.bins_completos(np.array([0.5]), bin_s=100.0,
                                                     inicio_s=0.0, fin_s=1.0)
        assert len(centros) == 0

    def test_bin_no_positivo_lanza(self):
        with pytest.raises(ValueError, match="positivo"):
            binning.bins_completos(np.array([0.5]), bin_s=0.0)

    def test_error_de_un_bin_vacio_no_es_cero(self):
        _, _, errores = binning.tasas_por_bin(np.array([0.5, 2.5]), bin_s=1.0,
                                              inicio_s=0.0, fin_s=3.0)
        assert errores[1] > 0

    def test_media_movil_de_una_constante_es_la_constante(self):
        valores = np.full(20, 3.0)
        medias, desviaciones = binning.media_movil(valores, 5)
        np.testing.assert_allclose(medias, 3.0)
        np.testing.assert_allclose(desviaciones, 0.0, atol=1e-12)


class TestElBinningNoAfectaALaFisica:
    """El defecto que se corrige: --bin-seconds movia la tasa publicada."""

    def test_la_tasa_no_depende_del_tamano_de_bin(self):
        corrida = io.load_run(CRUDOS / "000deg")
        tasa = rates.calcular_tasa(corrida)
        # La tasa se calcula sobre la ventana completa y no admite parametro de
        # bin: no hay forma de que un ajuste cosmetico la mueva.
        assert tasa.ventana_s == pytest.approx(corrida.ventana_s)
        assert "bin" not in rates.calcular_tasa.__code__.co_varnames
