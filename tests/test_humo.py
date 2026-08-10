"""Prueba de humo de muon-angle-analysis.

Existe desde el primer dia para que la puerta de verificacion sea ejecutable
antes de que haya solver que verificar. Se sustituye por la comprobacion real en
cuanto la haya: por orden de preferencia, una solucion analitica, una cantidad
conservada, o un estudio de convergencia.
"""


def test_el_paquete_se_importa():
    import sim.core  # noqa: F401
