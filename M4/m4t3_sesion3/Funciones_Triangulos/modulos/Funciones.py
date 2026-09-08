import pandas as pd
import numpy as np



def cargar_base(ruta_csv: str) -> pd.DataFrame:
    """
    Carga la base de datos del triángulo desde un archivo CSV.
    """
    try:
        return pd.read_csv(ruta_csv)
    except FileNotFoundError:
        print(f"Archivo no encontrado en la ruta: {ruta_csv}")
        return None

def cambiar_temporalidad(triangulo: pd.DataFrame,
                         columna_accidente: str = "accident_period",
                         columna_desarrollo: str = "development_period",
                         temporalidad: str = "monthly",
                         acumulativo: bool = True) -> pd.DataFrame:
    """
    Cambia la temporalidad de los periodos de accidente y desarrollo.
    temporalidad: 'monthly', 'quarterly', 'yearly'
    """
    # Validaciones
    for col in [columna_accidente, columna_desarrollo]:
        if col not in triangulo.columns:
            raise ValueError(f"La columna '{col}' no existe en el DataFrame")

    # Accident period → fechas
    triangulo[columna_accidente] = pd.to_datetime(triangulo[columna_accidente])

    if temporalidad == "monthly":
        triangulo[columna_accidente] = triangulo[columna_accidente].dt.to_period("M")
        triangulo[columna_desarrollo] = triangulo[columna_desarrollo].astype(int) // 1
    elif temporalidad == "quarterly":
        triangulo[columna_accidente] = triangulo[columna_accidente].dt.to_period("Q")
        triangulo[columna_desarrollo] = triangulo[columna_desarrollo].astype(int) // 3
    elif temporalidad == "yearly":
        triangulo[columna_accidente] = triangulo[columna_accidente].dt.to_period("Y")
        triangulo[columna_desarrollo] = triangulo[columna_desarrollo].astype(int) // 12
    else:
        raise ValueError("Temporalidad no reconocida. Usa: 'monthly', 'quarterly' o 'yearly'")

    if acumulativo:
        triangulo = (
        triangulo.sort_values(by=[columna_accidente, columna_desarrollo])
        .groupby([columna_accidente, columna_desarrollo])["amount"]
        .last()
        .reset_index()
    )

    return triangulo


def convertir_triangulo(triangulo: pd.DataFrame,concept = "Loss Incurred", index_col = "accident_period", columns_col = "development_period", values_col = "amount") -> pd.DataFrame:
    """
    Convierte la base de datos del triángulo a formato ancho (pivot).
    """
    triangulo_ancho = triangulo[triangulo["concept"] == concept].pivot_table(
        index=index_col,
        columns=columns_col,
        values=values_col,
        aggfunc="sum"
    ).sort_index(axis=0).sort_index(axis=1)
    return triangulo_ancho


def factores_desarrollo(triangulo_ancho, metodo="simple"):
    """
    Calcula factores de desarrollo a partir del triángulo.
    Métodos disponibles:
        - "simple": promedio simple
        - "ponderado": ponderado por tamaño de siniestro
        - "excluir_extremos": excluye máximo y mínimo antes de promediar
    """
    factores = []
    for col in range(triangulo_ancho.shape[1] - 1):
        num = triangulo_ancho.iloc[:, col+1]
        den = triangulo_ancho.iloc[:, col]
        ratios = num / den

        if metodo == "simple":
            factor = ratios.mean()
        elif metodo == "ponderado":
            factor = (num.sum() / den.sum())
        elif metodo == "excluir_extremos":
            factor = ratios.sort_values()[1:-1].mean()
        else:
            raise ValueError("Método no reconocido")

        factores.append(factor)
    return factores
        
"""
Clase profesional para calcular la reserva IBNR (Incurred But Not Reported)
a partir de un triángulo de desarrollo de siniestros.

- Funciones:
    - Cargar base de datos de triángulo.
    - Convertir la base de datos a un formato adecuado para el cálculo.
    - calcular los factores de desarrollo con promedios simples, ponderados y excluyendo máximo y mínimo.
    - Quitar periodos paraticulares para la selección de los factores de desarrollo.
    - Calcular la reserva IBNR a partir de los factores de desarrollo y el triángulo de desarrollo.
"""
