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

def convertir_triangulo(
    triangulo: pd.DataFrame,
    concept: str = "Loss Incurred",
    index_col: str = "accident_period",
    columns_col: str = "development_period",
    values_col: str = "amount",
) -> pd.DataFrame:
    """
    Convierte la base del triángulo a formato ancho.

    Filtra por concept y genera:


    development_period: 

        accident_period         0      1      2
        2016                    150     200    300
        2017                    200    250     320

    """

    columnas_requeridas = [
        concept,
        index_col,
        columns_col,
        values_col,
    ]

    for col in [index_col, columns_col, values_col, "concept"]:
        if col not in triangulo.columns:
            raise ValueError(
                f"La columna '{col}' no existe en el DataFrame."
            )

    df = triangulo[
        triangulo["concept"] == concept
    ].copy()

    if df.empty:
        raise ValueError(
            f"No existen registros para concept='{concept}'."
        )

    triangulo_ancho = (
        df.pivot_table(
            index=index_col,
            columns=columns_col,
            values=values_col,
            aggfunc="sum",
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    triangulo_ancho.columns.name = None

    return triangulo_ancho


def calcular_factores_desarrollo(
    triangulo: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula los factores de desarrollo individuales (LDF).

    Para cada período de desarrollo:

        LDF_j = C_(j+1) / C_j

    Ejemplo:

        Triángulo:

                    0      1      2      3
        2016      100    150    180    200
        2017      120    180    220    NaN

        Factores:

                    0-1    1-2    2-3
        2016       1.50   1.20   1.11
        2017       1.50   1.22    NaN
    """

    if not isinstance(triangulo, pd.DataFrame):
        raise TypeError(
            "triangulo debe ser un pandas.DataFrame."
        )

    if triangulo.shape[1] < 2:
        raise ValueError(
            "El triángulo debe tener al menos dos períodos "
            "de desarrollo."
        )

    df = triangulo.copy()

    # Asegurar orden cronológico del desarrollo
    df = df.sort_index(axis=1)

    # C_(j+1) / C_j
    factores = df.shift(-1, axis=1).div(df)

    # El último período no tiene factor
    factores = factores.iloc[:, :-1]

    # Nombres más explícitos
    factores.columns = [
        f"{col}_to_{df.columns[i + 1]}"
        for i, col in enumerate(factores.columns)
    ]

    return factores


def limpiar_factores(
    factores: pd.DataFrame,
    num_std: float | None = None,
    periodos_std: int | None = None,
    excluir_celdas: list[tuple] | None = None,
    excluir_periodos: list | None = None,
) -> pd.DataFrame:
    """
    Limpia una matriz de factores LDF.

    La función permite:

    1. Excluir períodos de accidente completos.
    2. Excluir celdas específicas.
    3. Excluir factores considerados atípicos mediante
       desviaciones estándar.

    Parámetros
    ----------
    factores : pd.DataFrame
        Matriz de factores LDF.

        Ejemplo:

                0_to_1  1_to_2  2_to_3
        2016      1.50    1.20    1.10
        2017      1.40    1.30     NaN
        2018      1.60     NaN      NaN

    num_std : float, opcional
        Número de desviaciones estándar utilizado para detectar
        atípicos.

        Ejemplo:
            num_std=2

        significa:

            media - 2*std <= factor <= media + 2*std

    periodos_std : int, opcional
        Número máximo de períodos recientes utilizados para
        calcular la media y desviación estándar.

        Por ejemplo:

            periodos_std=24

        utiliza hasta los últimos 24 períodos disponibles.

    excluir_celdas : list[tuple], opcional
        Lista de tuplas:

            [(periodo_accidente, factor)]

        Ejemplo:

            [
                (2019, "12_to_24"),
                (2021, "24_to_36")
            ]

    excluir_periodos : list, opcional
        Lista de períodos de accidente completos a excluir.

        Ejemplo:

            [2018, 2019]

    Retorna
    -------
    pd.DataFrame
        Matriz de factores limpia.
    """

    # =========================================================
    # 1. Validaciones
    # =========================================================

    if not isinstance(factores, pd.DataFrame):
        raise TypeError(
            "factores debe ser un pandas.DataFrame."
        )

    if num_std is not None:
        if not isinstance(num_std, (int, float)):
            raise TypeError(
                "num_std debe ser numérico."
            )

        if num_std <= 0:
            raise ValueError(
                "num_std debe ser mayor que 0."
            )

    if periodos_std is not None:
        if not isinstance(periodos_std, int):
            raise TypeError(
                "periodos_std debe ser un entero."
            )

        if periodos_std <= 0:
            raise ValueError(
                "periodos_std debe ser mayor que 0."
            )

    # =========================================================
    # 2. Copia
    # =========================================================

    factores_limpios = factores.copy()

    # =========================================================
    # 3. Excluir períodos completos
    # =========================================================

    if excluir_periodos is not None:

        for periodo in excluir_periodos:

            if periodo in factores_limpios.index:
                factores_limpios.loc[periodo, :] = np.nan

    # =========================================================
    # 4. Excluir celdas específicas
    # =========================================================

    if excluir_celdas is not None:

        for periodo, factor in excluir_celdas:

            if periodo not in factores_limpios.index:
                raise ValueError(
                    f"El período '{periodo}' no existe "
                    "en el índice del triángulo."
                )

            if factor not in factores_limpios.columns:
                raise ValueError(
                    f"El factor '{factor}' no existe "
                    "en las columnas."
                )

            factores_limpios.loc[periodo, factor] = np.nan

    # =========================================================
    # 5. Detectar atípicos por desviación estándar
    # =========================================================

    if num_std is not None:

        def detectar_atipicos(columna: pd.Series) -> pd.Series:

            # Valores disponibles
            validos = columna.dropna()

            # -------------------------------------------------
            # Seleccionar muestra para calcular media y std
            # -------------------------------------------------

            if periodos_std is not None:
                muestra = validos.tail(periodos_std)
            else:
                muestra = validos

            # Necesitamos suficientes observaciones
            if len(muestra) < 3:
                return columna

            # -------------------------------------------------
            # Estadísticos
            # -------------------------------------------------

            media = muestra.mean()

            desviacion = muestra.std(
                ddof=1
            )

            # -------------------------------------------------
            # Límites
            # -------------------------------------------------

            limite_inferior = (
                media - num_std * desviacion
            )

            limite_superior = (
                media + num_std * desviacion
            )

            # -------------------------------------------------
            # Excluir atípicos
            # -------------------------------------------------

            return columna.mask(
                (columna < limite_inferior)
                | (columna > limite_superior)
            )

        factores_limpios = factores_limpios.apply(
            detectar_atipicos,
            axis=0,
        )

    return factores_limpios

def seleccionar_periodos_ldf(
    factores_limpios: pd.DataFrame,
    ultimos_n: int | None = None,
) -> pd.DataFrame:
    """
    Selecciona los últimos N períodos disponibles para cada LDF.

    La selección se hace de forma independiente para cada columna
    de factores.

    Parámetros
    ----------
    factores_limpios : pd.DataFrame
        Matriz de factores después de aplicar exclusiones y
        tratamiento de atípicos.

    ultimos_n : int, opcional
        Número de períodos recientes a utilizar.

        Si es None, se utilizan todos los períodos disponibles.

    Retorna
    -------
    pd.DataFrame
        Matriz que conserva únicamente los factores seleccionados.
        Los factores no seleccionados se convierten en NaN.
    """

    if not isinstance(factores_limpios, pd.DataFrame):
        raise TypeError(
            "factores_limpios debe ser un pandas.DataFrame."
        )

    if ultimos_n is not None:
        if not isinstance(ultimos_n, int):
            raise TypeError(
                "ultimos_n debe ser un entero."
            )

        if ultimos_n <= 0:
            raise ValueError(
                "ultimos_n debe ser mayor que 0."
            )

    factores_seleccionados = factores_limpios.copy()

    # ---------------------------------------------------------
    # Ordenar cronológicamente
    # ---------------------------------------------------------

    factores_seleccionados = (
        factores_seleccionados.sort_index()
    )

    # ---------------------------------------------------------
    # Selección independiente para cada LDF
    # ---------------------------------------------------------

    for columna in factores_seleccionados.columns:

        validos = (
            factores_seleccionados[columna]
            .dropna()
        )

        if ultimos_n is not None:
            seleccionados = validos.tail(ultimos_n)
        else:
            seleccionados = validos

        # Todos los períodos que no pertenecen a la selección
        no_seleccionados = validos.index.difference(
            seleccionados.index
        )

        factores_seleccionados.loc[
            no_seleccionados,
            columna
        ] = np.nan

    return factores_seleccionados


def calcular_seleccion_ldf(
    triangulo: pd.DataFrame,
    factores_seleccionados: pd.DataFrame,
    metodo: str = "simple",
    columna_desarrollo: str = "development_period",
) -> pd.Series:
    """
    Calcula la selección de LDF para cada período de desarrollo.

    Métodos disponibles:

        "simple"
        "ponderado"
        "simple_sin_extremos"
        "ponderado_sin_extremos"

    Parámetros
    ----------
    triangulo : pd.DataFrame
        Triángulo acumulado en formato ancho.

        Ejemplo:

                    0      1      2      3
            2018   100    150    180    200
            2019   120    180    220    NaN
            2020   130    190    NaN    NaN

    factores_seleccionados : pd.DataFrame
        Matriz de factores después de aplicar:
            - exclusiones
            - eliminación de atípicos
            - selección de últimos N períodos

    metodo : str
        Método de cálculo:

            "simple"
            "ponderado"
            "simple_sin_extremos"
            "ponderado_sin_extremos"

    columna_desarrollo : str
        Nombre de la dimensión de desarrollo.

    Retorna
    -------
    pd.Series
        LDF seleccionado para cada período de desarrollo.
    """

    # =========================================================
    # 1. Validaciones
    # =========================================================

    if not isinstance(triangulo, pd.DataFrame):
        raise TypeError(
            "triangulo debe ser un pandas.DataFrame."
        )

    if not isinstance(factores_seleccionados, pd.DataFrame):
        raise TypeError(
            "factores_seleccionados debe ser un pandas.DataFrame."
        )

    metodos_validos = {
        "simple",
        "ponderado",
        "simple_sin_extremos",
        "ponderado_sin_extremos",
    }

    if metodo not in metodos_validos:
        raise ValueError(
            f"Método '{metodo}' no válido. "
            f"Usa uno de: {sorted(metodos_validos)}"
        )

    # =========================================================
    # 2. Copias y orden
    # =========================================================

    triangulo = triangulo.copy()
    factores = factores_seleccionados.copy()

    triangulo = triangulo.sort_index()
    factores = factores.sort_index()

    # =========================================================
    # 3. Determinar si se eliminan extremos
    # =========================================================

    sin_extremos = metodo in {
        "simple_sin_extremos",
        "ponderado_sin_extremos",
    }

    ponderado = metodo in {
        "ponderado",
        "ponderado_sin_extremos",
    }

    # =========================================================
    # 4. Calcular LDF columna por columna
    # =========================================================

    resultado = {}

    for i, factor_columna in enumerate(factores.columns):

        # -----------------------------------------------------
        # Factor correspondiente:
        #
        # C_(j+1) / C_j
        # -----------------------------------------------------

        desarrollo_inicial = triangulo.columns[i]
        desarrollo_siguiente = triangulo.columns[i + 1]

        factores_columna = factores[
            factor_columna
        ].dropna()

        # -----------------------------------------------------
        # Si no existen observaciones
        # -----------------------------------------------------

        if len(factores_columna) == 0:
            resultado[factor_columna] = np.nan
            continue

        # -----------------------------------------------------
        # Índices inicialmente seleccionados
        # -----------------------------------------------------

        indices = factores_columna.index

        # -----------------------------------------------------
        # Excluir máximo y mínimo
        # -----------------------------------------------------

        if sin_extremos and len(factores_columna) > 2:

            indice_min = factores_columna.idxmin()
            indice_max = factores_columna.idxmax()

            indices = indices[
                (indices != indice_min)
                & (indices != indice_max)
            ]

        # Si hay 1 o 2 observaciones no eliminamos nada.
        elif sin_extremos:
            indices = factores_columna.index

        # -----------------------------------------------------
        # Datos finales
        # -----------------------------------------------------

        factores_finales = factores_columna.loc[indices]

        if len(factores_finales) == 0:
            resultado[factor_columna] = np.nan
            continue

        # =====================================================
        # MÉTODO SIMPLE
        # =====================================================

        if not ponderado:

            resultado[factor_columna] = (
                factores_finales.mean()
            )

        # =====================================================
        # MÉTODO PONDERADO
        # =====================================================

        else:

            # Los mismos períodos utilizados en el factor
            # deben utilizarse en el numerador y denominador.

            valores_iniciales = triangulo.loc[
                indices,
                desarrollo_inicial,
            ]

            valores_finales = triangulo.loc[
                indices,
                desarrollo_siguiente,
            ]

            # Eliminar posibles NaN
            validos = (
                valores_iniciales.notna()
                & valores_finales.notna()
            )

            valores_iniciales = valores_iniciales[validos]
            valores_finales = valores_finales[validos]

            if len(valores_iniciales) == 0:
                resultado[factor_columna] = np.nan
                continue

            denominador = valores_iniciales.sum()
            numerador = valores_finales.sum()

            if denominador == 0:
                resultado[factor_columna] = np.nan
            else:
                resultado[factor_columna] = (
                    numerador / denominador
                )

    return pd.Series(
        resultado,
        name=f"ldf_{metodo}",
    )

def seleccionar_ldf(
    triangulo: pd.DataFrame,
    concept: str = "Loss Incurred",
    metodo: str = "simple",
    ultimos_n: int | None = None,
    num_std: float | None = None,
    periodos_std: int | None = None,
    excluir_periodos: list | None = None,
    excluir_celdas: list[tuple] | None = None,
) -> pd.DataFrame:
    """
    Ejecuta el proceso completo de selección de LDF.

    Flujo:

        Base
          ↓
        Triángulo ancho
          ↓
        Factores individuales
          ↓
        Exclusiones / atípicos
          ↓
        Selección de últimos N períodos
          ↓
        LDF seleccionado

    Parámetros
    ----------
    triangulo : pd.DataFrame
        Base original del triángulo.

    concept : str
        Concepto sobre el que se calcularán los LDF.

    metodo : str
        Método de selección:

            "simple"
            "ponderado"
            "simple_sin_extremos"
            "ponderado_sin_extremos"

    ultimos_n : int, opcional
        Número de períodos utilizados para la selección.

        None = todos los períodos disponibles.

    num_std : float, opcional
        Número de desviaciones estándar para identificar
        factores atípicos.

        None = no se eliminan atípicos.

    periodos_std : int, opcional
        Número de períodos recientes utilizados para calcular
        la media y desviación estándar de los atípicos.

        None = se utilizan todos los períodos disponibles.

    excluir_periodos : list, opcional
        Períodos de accidente completos que deben excluirse.

    excluir_celdas : list[tuple], opcional
        Celdas específicas que deben excluirse.

        Ejemplo:

            [
                (2018, "0_to_1"),
                (2020, "1_to_2")
            ]

    Retorna
    -------
    pd.DataFrame
        Tabla con los LDF seleccionados.

        Ejemplo:

                    LDF
        0_to_1      1.25
        1_to_2      1.15
        2_to_3      1.08
    """

    # =========================================================
    # 1. Convertir base a formato ancho
    # =========================================================

    triangulo_ancho = convertir_triangulo(
        triangulo=triangulo,
        concept=concept,
    )

    # =========================================================
    # 2. Calcular factores individuales
    # =========================================================

    factores = calcular_factores_desarrollo(
        triangulo=triangulo_ancho,
    )

    # =========================================================
    # 3. Limpiar factores
    # =========================================================

    factores_limpios = limpiar_factores(
        factores=factores,
        num_std=num_std,
        periodos_std=periodos_std,
        excluir_celdas=excluir_celdas,
        excluir_periodos=excluir_periodos,
    )

    # =========================================================
    # 4. Seleccionar últimos N períodos
    # =========================================================

    factores_seleccionados = seleccionar_periodos_ldf(
        factores_limpios=factores_limpios,
        ultimos_n=ultimos_n,
    )

    # =========================================================
    # 5. Calcular LDF seleccionado
    # =========================================================

    ldf = calcular_seleccion_ldf(
        triangulo=triangulo_ancho,
        factores_seleccionados=factores_seleccionados,
        metodo=metodo,
    )

    # =========================================================
    # 6. Convertir a DataFrame
    # =========================================================

    resultado = ldf.to_frame(
        name="LDF"
    )

    return resultado