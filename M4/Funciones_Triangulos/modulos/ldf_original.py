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

def cambiar_temporalidad(
    triangulo: pd.DataFrame,
    columna_accidente: str = "accident_period",
    columna_desarrollo: str = "development_period",
    columna_valor: str = "amount",
    temporalidad: str = "monthly",
    temporalidad_nueva: str = "yearly",
    acumulativo: bool = True
) -> pd.DataFrame:
    """
    Cambia la temporalidad de los periodos de accidente y desarrollo.
    temporalidad: 'monthly', 'quarterly', 'yearly' (temporalidad original)
    temporalidad_nueva: 'monthly', 'quarterly', 'yearly' (temporalidad destino)
    acumulativo: si True, se queda con el último valor de cada año de
                desarrollo y suma todos los accident_period dentro del bloque.
    """
    # Validaciones
    for col in [columna_accidente, columna_desarrollo, columna_valor]:
        if col not in triangulo.columns:
            raise ValueError(f"La columna '{col}' no existe en el DataFrame")

    # Convertimos accident_period a datetime
    triangulo[columna_accidente] = pd.to_datetime(
        triangulo[columna_accidente]
    )

    # Ajustamos accident_period a la nueva temporalidad
    if temporalidad_nueva == "monthly":
        triangulo[columna_accidente] = (
            triangulo[columna_accidente].dt.to_period("M")
        )
        triangulo[columna_desarrollo] = (
            triangulo[columna_desarrollo] // 1
        )

    elif temporalidad_nueva == "quarterly":
        triangulo[columna_accidente] = (
            triangulo[columna_accidente].dt.to_period("Q")
        )
        triangulo[columna_desarrollo] = (
            triangulo[columna_desarrollo] // 3
        )

    elif temporalidad_nueva == "yearly":
        triangulo[columna_accidente] = (
            triangulo[columna_accidente].dt.to_period("Y")
        )
        triangulo[columna_desarrollo] = (
            triangulo[columna_desarrollo] // 12
        )

    else:
        raise ValueError(
            "Temporalidad nueva no reconocida. "
            "Usa: 'monthly', 'quarterly' o 'yearly'"
        )

    # Si es acumulativo → último valor de cada desarrollo
    # y suma de accident_period
    if acumulativo:
        triangulo_final = (
            triangulo
            .sort_values(
                by=[
                    columna_accidente,
                    columna_desarrollo,
                    "concept"
                ]
            )
            .groupby(
                [
                    columna_accidente,
                    columna_desarrollo,
                    "concept"
                ]
            )[columna_valor]
            .last()
            .groupby(level=1)
            .sum()
            .reset_index()
        )

    else:
        triangulo_final = triangulo

    return triangulo_final

def convertir_triangulo(
    triangulo: pd.DataFrame,
    concept="Loss Incurred",
    index_col="accident_period",
    columns_col="development_period",
    values_col="amount"
    ) -> pd.DataFrame:
    """
    Convierte la base de datos del triángulo a formato ancho (pivot).
    """
    triangulo_ancho = (
    triangulo[
        triangulo["concept"] == concept
    ]
    .pivot_table(
        index=index_col,
        columns=columns_col,
        values=values_col,
        aggfunc="sum"
    )
    .sort_index(axis=0)
    .sort_index(axis=1)
)

    return triangulo_ancho

def calcular_factores_desarrollo(
df_triangulo,
col_periodo="accident_period"
):
    """
    Calcula la matriz de factores de desarrollo individuales.
    """

    df = df_triangulo.copy()

    if col_periodo in df.columns:
        df = df.set_index(col_periodo)

    df_factores = df.shift(-1, axis=1) / df

    return df_factores.iloc[:, :-1]


def limpiar_factores(
    df_factores,
    num_std=None,
    periodos_std=None,
    excluir_celdas=None,
    excluir_periodos=None,
    ):
    """
    Aplica filtros de exclusión a la matriz de factores de desarrollo.
    """
    factores_limpios = df_factores.copy()

    # 1. Excluir períodos completos (filas)
    if excluir_periodos:
        factores_limpios.loc[
            factores_limpios.index.isin(excluir_periodos)
        ] = np.nan

    # 2. Excluir celdas específicas manualmente
    if excluir_celdas:
        for periodo, col in excluir_celdas:
            if (
                periodo in factores_limpios.index
                and col in factores_limpios.columns
            ):
                factores_limpios.loc[periodo, col] = np.nan

    # 3. Excluir outliers por Desviación Estándar
    # usando 'periodos_std' (M períodos)
    if num_std is not None and num_std > 0:

        def _filtrar_std(col):

            validos = col.dropna()

            # Tomar los últimos M períodos válidos para
            # la muestra de desviación estándar
            muestra_std = (
                validos.tail(periodos_std)
                if (
                    periodos_std
                    and isinstance(periodos_std, int)
                )
                else validos
            )

            if len(muestra_std) < 3:
                return col

            media = muestra_std.mean()
            std = muestra_std.std()

            lim_inferior = media - (num_std * std)
            lim_superior = media + (num_std * std)

            # Enmascarar atípicos en toda la columna
            # según la banda de tolerancia calculada
            return col.mask(
                (col < lim_inferior)
                | (col > lim_superior)
            )

        factores_limpios = factores_limpios.apply(
            _filtrar_std,
            axis=0
        )

    return factores_limpios

def resumen_factores_desarrollo(
    df_triangulo,
    col_periodo="accident_period",
    metodo="ponderado",
    ultimos_n=None,
    num_std=None,
    periodos_std=None,
    excluir_celdas=None,
    excluir_periodos=None,
    ):
    """
    Calcula los factores agregados permitiendo una ventana
    para atípicos (periodos_std) diferente a la ventana
    de promediación (ultimos_n).
    """

    df = df_triangulo.copy()

    if col_periodo in df.columns:
        df = df.set_index(col_periodo)

    df = df.sort_index()

    # 1. Matriz de factores inicial
    df_factores = calcular_factores_desarrollo(
        df,
        col_periodo=None
    )

    # 2. Aplicar limpiezas
    # (atípicos calculados sobre periodos_std)
    df_factores = limpiar_factores(
        df_factores,
        num_std=num_std,
        periodos_std=(
            periodos_std
            if periodos_std is not None
            else ultimos_n
        ),
        excluir_celdas=excluir_celdas,
        excluir_periodos=excluir_periodos,
    )

    def _obtener_ultimas_n_validas(serie, n):

        validas = serie.dropna()

        if n is not None and isinstance(n, int):
            return validas.tail(n)

        return validas

    # -------------------------------------------------------------
    # Promedios finales sobre las últimas 'n' observaciones limpias
    # -------------------------------------------------------------

    # 1. Promedio Simple
    def _promedio_simple_n(col):

        datos = _obtener_ultimas_n_validas(
            col,
            ultimos_n
        )

        return (
            datos.mean()
            if len(datos) > 0
            else np.nan
        )

    prom_simple = df_factores.apply(
        _promedio_simple_n,
        axis=0
    )

    # 2. Promedio Ponderado
    prom_ponderado_dict = {}

    for i in range(len(df_factores.columns)):

        col_nombre = df_factores.columns[i]

        factores_col = df_factores[col_nombre]

        indices_validos = _obtener_ultimas_n_validas(
            factores_col,
            ultimos_n
        ).index

        if len(indices_validos) > 0:

            suma_t = df.loc[
                indices_validos,
                df.columns[i]
            ].sum()

            suma_t_mas_1 = df.loc[
                indices_validos,
                df.columns[i + 1]
            ].sum()

            prom_ponderado_dict[col_nombre] = (
                suma_t_mas_1 / suma_t
                if suma_t != 0
                else np.nan
            )

        else:
            prom_ponderado_dict[col_nombre] = np.nan

    prom_ponderado = pd.Series(
        prom_ponderado_dict
    )

    # 3. Excluir Extremos (Min / Max)
    def _promedio_sin_extremos(col):

        validos = _obtener_ultimas_n_validas(
            col,
            ultimos_n
        )

        if len(validos) <= 2:

            return (
                validos.mean()
                if len(validos) > 0
                else np.nan
            )

        validos_filtrados = validos.drop(
            [
                validos.idxmax(),
                validos.idxmin()
            ]
        )

        return validos_filtrados.mean()

    prom_excluir_extremos = df_factores.apply(
        _promedio_sin_extremos,
        axis=0
    )

    resúmenes = {
        "simple": prom_simple,
        "ponderado": prom_ponderado,
        "excluir_extremos": prom_excluir_extremos,
    }

    if metodo in resúmenes:

        nombre_filas = (
            f"factor_{metodo}_u{ultimos_n}"
            if ultimos_n
            else f"factor_{metodo}"
        )

        return pd.DataFrame(
            [resúmenes[metodo]],
            index=[nombre_filas]
        )

    elif metodo == "todos":

        return pd.DataFrame(resúmenes).T

    else:

        raise ValueError(
            f"Método '{metodo}' no válido."
        )