"""Tests de los modelos y de la inferencia.

El primero es un test de REGRESION: fija el defecto que motivo reconstruir el
analisis entero. Si alguna vez vuelve a fallar, el repositorio ha vuelto a
publicar un exponente que es redondeo de doble precision.
"""

import numpy as np
import pytest

from lib_muon import models


class TestRegresionCoseno90:
    """cos(deg2rad(90)) no es cero y por poco no se nota."""

    def test_numpy_no_da_cero_en_90_grados(self):
        crudo = np.cos(np.deg2rad(90.0))
        assert crudo != 0.0
        assert crudo == pytest.approx(6.123233995736766e-17, rel=1e-12)

    def test_el_anclaje_si_da_cero_exacto(self):
        anclado = models.coseno_anclado(np.array([90.0]))
        assert anclado[0] == 0.0

    def test_el_bug_original_reproducido(self):
        """Sin anclar, cos(90)^n con n pequeno vale del orden de 1."""
        crudo = np.cos(np.deg2rad(90.0))
        assert crudo**0.01 == pytest.approx(0.688, abs=1e-3)
        assert crudo**0.006 == pytest.approx(0.799, abs=1e-3)

    def test_anclado_cualquier_exponente_positivo_da_cero(self):
        cos_t = models.coseno_anclado(np.array([0.0, 45.0, 90.0]))
        for n in (0.006, 0.01, 0.5, 1.0, 2.0, 5.0):
            forma = models.forma_cosn(cos_t, n)
            assert forma[-1] == 0.0, f"cos^{n}(90 grados) deberia ser cero exacto"
        assert models.forma_cosn(cos_t, 2.0)[0] == pytest.approx(1.0)
        assert models.forma_cosn(cos_t, 2.0)[1] == pytest.approx(0.5)

    def test_los_angulos_normales_no_se_ven_afectados(self):
        angulos = np.array([0.0, 5.0, 15.0, 25.0, 35.0, 45.0, 89.9])
        anclado = models.coseno_anclado(angulos)
        crudo = np.cos(np.deg2rad(angulos))
        np.testing.assert_allclose(anclado, crudo, rtol=0, atol=0)

    def test_ajuste_de_cosn_puro_es_imposible_con_cuentas_a_90_grados(self):
        """La consecuencia del anclaje: verosimilitud rigurosamente nula."""
        cos_t = models.coseno_anclado(np.array([0.0, 45.0, 90.0]))
        tiempos = np.array([300.0, 300.0, 300.0])
        cuentas = np.array([600.0, 550.0, 450.0])
        ajuste = models.ajustar_cosn_puro(cuentas, tiempos, cos_t)
        assert not np.isfinite(ajuste.nll)
        assert not ajuste.convergio


class TestVerosimilitud:
    def test_nll_infinito_si_mu_cero_con_cuentas(self):
        assert models.nll_poisson(np.array([0.0]), np.array([5.0])) == np.inf

    def test_nll_finito_si_mu_cero_sin_cuentas(self):
        assert np.isfinite(models.nll_poisson(np.array([0.0]), np.array([0.0])))

    def test_minimo_de_la_verosimilitud_en_el_valor_verdadero(self):
        """Para una Poisson, -logL se minimiza en mu = N."""
        cuentas = np.array([100.0])
        valores = np.array([80.0, 95.0, 100.0, 105.0, 120.0])
        nll = [models.nll_poisson(np.array([m]), cuentas) for m in valores]
        assert np.argmin(nll) == 2

    def test_constante_recupera_la_tasa_exacta(self):
        tiempos = np.array([100.0, 200.0, 300.0])
        cuentas = np.array([200.0, 400.0, 600.0])
        ajuste = models.ajustar_constante(cuentas, tiempos)
        assert ajuste.parametros["R_hz"] == pytest.approx(2.0)
        assert models.deviance(cuentas, ajuste.mu) == pytest.approx(0.0, abs=1e-9)


class TestRecuperacionDeParametros:
    """Sobre datos sinteticos con la respuesta conocida de antemano."""

    @pytest.mark.parametrize("n_verdadero", [0.5, 1.0, 2.0, 3.0])
    def test_recupera_un_n_conocido(self, n_verdadero):
        generador = np.random.default_rng(20260810)
        angulos = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
        cos_t = models.coseno_anclado(angulos)
        tiempos = np.full(len(angulos), 20000.0)
        tasa = 2.0
        mu = tasa * tiempos * models.forma_cosn(cos_t, n_verdadero)
        cuentas = generador.poisson(mu).astype(float)

        ajuste = models.ajustar_cosn_puro(cuentas, tiempos, cos_t)
        assert ajuste.convergio
        assert ajuste.parametros["n"] == pytest.approx(n_verdadero, abs=0.05)
        assert ajuste.parametros["R_hz"] == pytest.approx(tasa, rel=0.02)

    def test_recupera_senal_y_fondo_conocidos(self):
        generador = np.random.default_rng(11)
        angulos = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
        cos_t = models.coseno_anclado(angulos)
        tiempos = np.full(len(angulos), 40000.0)
        s_verdadero, b_verdadero, n_verdadero = 1.0, 0.5, 2.0
        mu = tiempos * (s_verdadero * models.forma_cosn(cos_t, n_verdadero) + b_verdadero)
        cuentas = generador.poisson(mu).astype(float)

        ajuste = models.ajustar_senal_fondo(cuentas, tiempos, cos_t, n=n_verdadero)
        assert ajuste.parametros["S_hz"] == pytest.approx(s_verdadero, rel=0.06)
        assert ajuste.parametros["B_hz"] == pytest.approx(b_verdadero, rel=0.10)

    def test_el_perfil_acota_el_valor_verdadero(self):
        generador = np.random.default_rng(7)
        angulos = np.array([0.0, 20.0, 40.0, 60.0, 80.0])
        cos_t = models.coseno_anclado(angulos)
        tiempos = np.full(len(angulos), 30000.0)
        n_verdadero = 2.0
        mu = 2.0 * tiempos * models.forma_cosn(cos_t, n_verdadero)
        cuentas = generador.poisson(mu).astype(float)
        ajuste = models.ajustar_cosn_puro(cuentas, tiempos, cos_t)

        def nll_de_n(n):
            forma = models.forma_cosn(cos_t, n)
            denominador = float(np.sum(tiempos * forma))
            r = float(cuentas.sum() / denominador)
            return models.nll_poisson(r * tiempos * forma, cuentas)

        bajo, alto = models.perfil_verosimilitud(
            nll_de_n, ajuste.parametros["n"], nll_de_n(ajuste.parametros["n"])
        )
        assert bajo < n_verdadero < alto


class TestComparacionDeModelos:
    def test_aic_penaliza_los_parametros(self):
        cuentas = np.array([100.0, 100.0, 100.0])
        tiempos = np.array([50.0, 50.0, 50.0])
        constante = models.ajustar_constante(cuentas, tiempos)
        cos_t = models.coseno_anclado(np.array([0.0, 10.0, 20.0]))
        senal = models.ajustar_senal_fondo(cuentas, tiempos, cos_t, n=2.0)
        # Los datos son planos: el modelo con mas parametros no gana nada
        # y el AIC lo tiene que penalizar.
        assert constante.aic < senal.aic

    def test_razon_de_verosimilitudes_no_ve_senal_donde_no_la_hay(self):
        d, p = models.razon_verosimilitudes(nll_nulo=100.0, nll_alternativo=99.9,
                                            gdl_extra=1)
        assert d == pytest.approx(0.2, abs=1e-9)
        assert p > 0.05

    def test_razon_de_verosimilitudes_ve_senal_grande(self):
        d, p = models.razon_verosimilitudes(nll_nulo=110.0, nll_alternativo=100.0,
                                            gdl_extra=1)
        assert p < 1e-4


class TestSistematico:
    def test_el_sistematico_relaja_el_chi2(self):
        tasas = np.array([2.0, 1.8, 2.2, 1.9, 2.1])
        errores = np.full(5, 0.05)
        _, chi2_sin, p_sin = models.chi2_con_sistematico(tasas, errores, 0.0)
        _, chi2_con, p_con = models.chi2_con_sistematico(tasas, errores, 0.10)
        assert chi2_con < chi2_sin
        assert p_con > p_sin

    def test_sistematico_necesario_es_cero_si_ya_es_compatible(self):
        tasas = np.array([2.0, 2.01, 1.99])
        errores = np.full(3, 0.5)
        assert models.sistematico_necesario(tasas, errores) == 0.0

    def test_sistematico_necesario_alcanza_el_p_objetivo(self):
        tasas = np.array([2.0, 1.6, 2.4, 1.8, 2.2])
        errores = np.full(5, 0.05)
        f = models.sistematico_necesario(tasas, errores, p_objetivo=0.05)
        assert f > 0
        _, _, p = models.chi2_con_sistematico(tasas, errores, f)
        assert p == pytest.approx(0.05, abs=1e-6)
