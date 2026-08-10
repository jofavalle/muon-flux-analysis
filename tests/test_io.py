"""Tests de lectura de los archivos crudos.

Corren sobre los datos reales del repositorio: son 256 kB y estan versionados
precisamente para que esto sea posible.
"""

from pathlib import Path

import numpy as np
import pytest

from lib_muon import io

RAIZ = Path(__file__).resolve().parents[1]
CRUDOS = RAIZ / "data" / "raw"
ANGULOS = [0, 5, 15, 25, 35, 45, 90, 360]


class TestParseoDeLinea:
    def test_linea_valida(self):
        linea = "1\t83\t201\t3741\t2.6\t73157.02\t23.13\t1411\t1\tCOSMIC"
        campos = io.parse_line(linea)
        assert campos[0] == 1
        assert campos[1] == 83
        assert campos[5] == pytest.approx(73157.02)
        assert campos[6] == pytest.approx(23.13)
        assert campos[8] == 1
        assert campos[9] == "COSMIC"

    def test_separacion_por_espacios_tambien_vale(self):
        linea = "1 83 201 3741 2.6 93609.2 27.8 1411 1 COSMIC"
        assert io.parse_line(linea)[0] == 1

    @pytest.mark.parametrize(
        "mala",
        [
            " 118  | Hexa address:  0x76",   # la de data/raw/015deg, literal
            "1\t83\t201",
            "",
            "no son numeros en absoluto aqui dentro nada de nada cero",
        ],
    )
    def test_lineas_malformadas_lanzan(self, mala):
        with pytest.raises(ValueError):
            io.parse_line(mala)

    def test_columnas_de_mas_tambien_lanzan(self):
        """El script original aceptaba >= 10 columnas y se quedaba con las 10
        primeras, de modo que una linea con basura pegada al final entraba
        como buena."""
        with pytest.raises(ValueError):
            io.parse_line("1 83 201 3741 2.6 93609.2 27.8 1411 1 COSMIC extra")


class TestLecturaDeArchivosReales:
    def test_los_ocho_archivos_se_leen(self):
        corridas = io.load_all(CRUDOS)
        assert sorted(corridas) == ANGULOS

    def test_recuento_total_de_eventos(self):
        corridas = io.load_all(CRUDOS)
        assert sum(len(c) for c in corridas.values()) == 4760

    def test_015deg_tiene_exactamente_una_linea_rechazada(self):
        """El archivo esta truncado por arriba: perdio las tres primeras
        lineas del preambulo y la cuarta quedo partida, de modo que su primera
        linea es el fragmento ' 118  | Hexa address:  0x76'. El script original
        la descartaba en silencio con `except Exception: continue`.

        La linea de nombres de columna si sobrevivio al truncado, asi que el
        archivo se lee sin ambiguedad; lo perdido es el preambulo del sensor.
        """
        corrida = io.load_run(CRUDOS / "015deg")
        assert corrida.calidad.lineas_rechazadas == 1
        assert corrida.calidad.textos_rechazados == ["118  | Hexa address:  0x76"]
        assert corrida.calidad.cabecera_presente

    def test_ningun_otro_archivo_tiene_lineas_rechazadas(self):
        for angulo in ANGULOS:
            if angulo == 15:
                continue
            corrida = io.load_run(CRUDOS / f"{angulo:03d}deg")
            assert corrida.calidad.lineas_rechazadas == 0, (
                f"{angulo} grados tiene lineas rechazadas inesperadas"
            )

    def test_la_cabecera_de_columnas_esta_en_los_ocho(self):
        for angulo in ANGULOS:
            corrida = io.load_run(CRUDOS / f"{angulo:03d}deg")
            assert corrida.calidad.cabecera_presente


class TestOrdenDeColumnas:
    def test_las_columnas_estan_invertidas_en_los_ocho_archivos(self):
        """La cabecera declara Temp[C] Pressure[Pa] y vienen al reves."""
        for angulo in ANGULOS:
            corrida = io.load_run(CRUDOS / f"{angulo:03d}deg")
            assert corrida.calidad.columnas_invertidas, (
                f"{angulo} grados: se esperaba presion antes que temperatura"
            )

    def test_presion_y_temperatura_quedan_en_rango_fisico(self):
        for angulo in ANGULOS:
            corrida = io.load_run(CRUDOS / f"{angulo:03d}deg")
            assert io.RANGO_PRESION_PA[0] < np.median(corrida.presion_pa) < io.RANGO_PRESION_PA[1]
            assert io.RANGO_TEMPERATURA_C[0] < np.median(corrida.temperatura_c) < io.RANGO_TEMPERATURA_C[1]
            assert corrida.calidad.filas_fuera_de_rango == 0

    def test_se_niega_a_adivinar_si_ninguna_columna_encaja(self):
        with pytest.raises(ValueError, match="no se puede decidir"):
            io._decidir_orden_columnas(np.array([5.0, 5.0]), np.array([7.0, 7.0]))


class TestArranqueDelSensor:
    def test_recuento_por_archivo(self):
        """El BMP280 emite 73157.02 Pa / 23.13 C antes de estabilizarse."""
        esperado = {0: 2, 5: 2, 15: 0, 25: 2, 35: 1, 45: 0, 90: 0, 360: 1}
        for angulo, cuantas in esperado.items():
            corrida = io.load_run(CRUDOS / f"{angulo:03d}deg")
            assert corrida.calidad.filas_arranque_sensor == cuantas, (
                f"{angulo} grados: se esperaban {cuantas} filas de arranque"
            )

    def test_siempre_al_principio_de_la_corrida(self):
        for angulo in ANGULOS:
            corrida = io.load_run(CRUDOS / f"{angulo:03d}deg")
            indices = corrida.calidad.indices_arranque_sensor
            if indices:
                assert max(indices) < 10, (
                    f"{angulo} grados: arranque del sensor fuera del principio"
                )


class TestTiempos:
    def test_el_reloj_nunca_retrocede(self):
        """Justifica usar la resta directa tN - t1 en vez de acumular
        diferencias positivas, que es lo que hacia el script original."""
        for angulo in ANGULOS:
            corrida = io.load_run(CRUDOS / f"{angulo:03d}deg")
            assert np.all(np.diff(corrida.timestamp_ms) > 0), (
                f"{angulo} grados: el reloj retrocede o se repite"
            )

    def test_fraccion_muerta_en_el_rango_esperado(self):
        for angulo in ANGULOS:
            corrida = io.load_run(CRUDOS / f"{angulo:03d}deg")
            assert 0.085 < corrida.fraccion_muerta < 0.10, (
                f"{angulo} grados: fraccion muerta {corrida.fraccion_muerta:.4f} "
                "fuera del 9 % caracteristico del conjunto"
            )

    def test_todas_las_corridas_caben_en_la_ventana_de_600_s(self):
        """El --max-ms de 600 s del script original no recortaba nada: la
        corrida mas larga dura 444 s."""
        for angulo in ANGULOS:
            corrida = io.load_run(CRUDOS / f"{angulo:03d}deg")
            assert corrida.ventana_s < 600.0


class TestCoincidencia:
    def test_el_filtro_de_coincidencia_es_un_no_op(self):
        """Coincident == 1 en los 4760 eventos. Sin singles registrados, la
        tasa de accidentales no es estimable desde estos datos."""
        corridas = io.load_all(CRUDOS)
        for angulo, corrida in corridas.items():
            assert np.all(corrida.coincidente == 1), (
                f"{angulo} grados tiene eventos no coincidentes"
            )
