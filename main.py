"""
main.py
Sesión 3 · Minería de Datos (NRC: 95400) - UNIMINUTO
Proyecto: Predicción de Costos Médicos
Integrantes: Jose Manuel Salgado, Juan Felipe Meneses Rodríguez

Punto de entrada único: ejecuta el pipeline de limpieza para los dos
datasets del proyecto. Gracias a que InsuranceCleaner y
MedicalPremiumCleaner comparten la misma interfaz (DataCleaner), este
archivo no necesita conocer los detalles internos de cada limpieza
(principio de sustitución de Liskov) y es trivial agregar un tercer
dataset en el futuro (principio abierto/cerrado): basta con crear su
clase en cleaners/ y añadirla a la lista de abajo.
"""

from limpieza.insurance_cleaner import InsuranceCleaner
from limpieza.medicalpremium_cleaner import MedicalPremiumCleaner

RUTA_INSURANCE_RAW = "data/raw/insurance.csv"
RUTA_INSURANCE_LIMPIO = "data/limpio/insurance_limpio.csv"

RUTA_PREMIUM_RAW = "data/raw/Medicalpremium.csv"
RUTA_PREMIUM_LIMPIO = "data/limpio/Medicalpremium_limpio.csv"


def main() -> None:
    pipelines = [
        InsuranceCleaner(RUTA_INSURANCE_RAW, RUTA_INSURANCE_LIMPIO),
        MedicalPremiumCleaner(RUTA_PREMIUM_RAW, RUTA_PREMIUM_LIMPIO),
    ]
    for pipeline in pipelines:
        pipeline.ejecutar()


if __name__ == "__main__":
    main()
