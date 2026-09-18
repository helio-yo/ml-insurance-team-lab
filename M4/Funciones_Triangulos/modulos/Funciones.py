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
    acumular: bool = False
) -> pd.DataFrame:
    """
    Convierte la base del triángulo a formato ancho.

    Filtra por concept y genera:

        Si ``acumular=True``, interpreta ``amount`` como incremental y
    acumula horizontalmente por período de accidente.

    
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

    if not isinstance(acumular, bool):
        raise TypeError(
            "acumular debe ser un booleano."
        )

    if acumular:
        triangulo_ancho = acumular_triangulo(
            triangulo_ancho
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
    acumular: bool = False
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

    acumular : bool, opcional
    Si es True, interpreta la base como incremental antes de
    calcular los factores de desarrollo.

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

def calcular_ratio(
    base: pd.DataFrame,
    columna_numerador: str,
    columna_denominador: str,
    columnas_grupo: list[str] | None = None,
    nombre_ratio: str = "ratio",
) -> pd.DataFrame:
    """
    Calcula ratios entre dos columnas numéricas de una base.

    Si se indican columnas_grupo, agrega el numerador y denominador
    antes de calcular el ratio.

    Parámetros
    ----------
    base : pd.DataFrame
        Base que contiene el numerador y el denominador.

    columna_numerador : str
        Columna que se usará como numerador.

    columna_denominador : str
        Columna que se usará como denominador.

    columnas_grupo : list[str], opcional
        Columnas con las que se agregará la base antes de calcular
        el ratio.

    nombre_ratio : str
        Nombre de la columna de resultado.

    Retorna
    -------
    pd.DataFrame
        Base original, o agregada, con la columna del ratio.
        Cuando el denominador es cero, el ratio se reporta como NaN.
    """

    # =========================================================
    # 1. Validaciones
    # =========================================================

    if not isinstance(base, pd.DataFrame):
        raise TypeError("base debe ser un pandas.DataFrame.")

    for columna in [columna_numerador, columna_denominador]:
        if columna not in base.columns:
            raise ValueError(
                f"La columna '{columna}' no existe en la base."
            )

    if columnas_grupo is not None:
        if not isinstance(columnas_grupo, list):
            raise TypeError(
                "columnas_grupo debe ser una lista o None."
            )

        for columna in columnas_grupo:
            if columna not in base.columns:
                raise ValueError(
                    f"La columna '{columna}' no existe en la base."
                )

    if not isinstance(nombre_ratio, str) or not nombre_ratio:
        raise ValueError(
            "nombre_ratio debe ser un texto no vacío."
        )

    # =========================================================
    # 2. Agregar, si aplica, y calcular ratio
    # =========================================================

    if columnas_grupo is None:
        resultado = base.copy()
    else:
        resultado = (
            base.groupby(
                columnas_grupo,
                dropna=False,
                as_index=False,
            )[[columna_numerador, columna_denominador]]
            .sum()
        )

    numerador = pd.to_numeric(
        resultado[columna_numerador],
        errors="raise",
    )
    denominador = pd.to_numeric(
        resultado[columna_denominador],
        errors="raise",
    )

    resultado[nombre_ratio] = numerador.div(
        denominador.replace(0, np.nan)
    )

    return resultado

def cortar_base_fecha_valuacion(
    base: pd.DataFrame,
    fecha_valuacion: str | pd.Timestamp,
    index_col: str = "accident_period",
    columns_col: str = "development_period",
    unidad_desarrollo: str = "meses",
) -> pd.DataFrame:
    """
    Corta una base de triángulo a una fecha de valuación.

    El período de accidente se interpreta como fecha de origen y el
    período de desarrollo como meses, días o años desde esa fecha.
    Sólo conserva observaciones cuya fecha de desarrollo ya ocurrió.

    Parámetros
    ----------
    base : pd.DataFrame
        Base de triángulo en formato largo.

    fecha_valuacion : str o pd.Timestamp
        Fecha máxima de valuación a conservar.

    index_col : str
        Columna que contiene la fecha de origen.

    columns_col : str
        Columna que contiene el período de desarrollo.

    unidad_desarrollo : str
        Unidad del período de desarrollo: "meses", "dias" o "años".

    Retorna
    -------
    pd.DataFrame
        Copia de la base limitada a la fecha de valuación.
    """

    # =========================================================
    # 1. Validaciones
    # =========================================================

    if not isinstance(base, pd.DataFrame):
        raise TypeError("base debe ser un pandas.DataFrame.")

    for columna in [index_col, columns_col]:
        if columna not in base.columns:
            raise ValueError(
                f"La columna '{columna}' no existe en la base."
            )

    unidades_validas = {
        "meses": "months",
        "dias": "days",
        "años": "years",
    }

    if unidad_desarrollo not in unidades_validas:
        raise ValueError(
            "unidad_desarrollo debe ser 'meses', 'dias' o 'años'."
        )

    fecha_valuacion = pd.to_datetime(
        fecha_valuacion,
        errors="raise",
    )

    if pd.isna(fecha_valuacion):
        raise ValueError(
            "fecha_valuacion no puede ser nula."
        )

    # =========================================================
    # 2. Calcular fecha de desarrollo
    # =========================================================

    resultado = base.copy()
    fechas_origen = pd.to_datetime(
        resultado[index_col],
        errors="raise",
    )
    periodos = pd.to_numeric(
        resultado[columns_col],
        errors="raise",
    )

    if periodos.isna().any():
        raise ValueError(
            "columns_col no puede contener valores nulos."
        )

    if not np.all(np.equal(periodos, np.floor(periodos))):
        raise ValueError(
            "Los períodos de desarrollo deben ser enteros."
        )

    nombre_unidad = unidades_validas[unidad_desarrollo]
    fechas_desarrollo = pd.Series(
        [
            fecha + pd.DateOffset(**{nombre_unidad: int(periodo)})
            for fecha, periodo in zip(fechas_origen, periodos)
        ],
        index=resultado.index,
    )

    # =========================================================
    # 3. Aplicar fecha de valuación
    # =========================================================

    conservar = (
        (fechas_origen <= fecha_valuacion)
        & (fechas_desarrollo <= fecha_valuacion)
    )

    return resultado.loc[conservar].copy()

def _convertir_a_serie_bf(
    valor: float | int | dict | pd.Series,
    indice: pd.Index,
    nombre: str,
) -> pd.Series:
    """Convierte un insumo escalar o por período en una serie."""

    if np.isscalar(valor):
        serie = pd.Series(valor, index=indice, dtype=float)
    elif isinstance(valor, dict):
        serie = pd.Series(valor, dtype=float).reindex(indice)
    elif isinstance(valor, pd.Series):
        serie = valor.astype(float).reindex(indice)
    else:
        raise TypeError(
            f"{nombre} debe ser escalar, dict o pandas.Series."
        )

    if serie.isna().any():
        raise ValueError(
            f"{nombre} debe tener un valor para cada período "
            "de accidente."
        )

    if (serie < 0).any():
        raise ValueError(
            f"{nombre} no puede contener valores negativos."
        )

    return serie

def calcular_bornhuetter_ferguson(
    triangulo: pd.DataFrame,
    ldf_seleccionados: pd.Series | pd.DataFrame,
    primas: float | int | dict | pd.Series,
    ratio_perdida_esperada: float | int | dict | pd.Series,
    factor_cola: float = 1.0,
    validar_monotonia: bool = True,
    permitir_cdf_menor_uno: bool = False,
) -> pd.DataFrame:
    """
    Calcula ultimate y reserva mediante Bornhuetter-Ferguson (B-F).

    B-F combina el valor acumulado observado con la porción no
    desarrollada de la pérdida esperada:

        CDF_j = LDF_j x ... x LDF_ultimo x factor_cola
        Ultimate_BF = Observado_j
                      + (Prima x Ratio esperado) x (1 - 1 / CDF_j)

    El CDF se construye desde los LDF seleccionados para la edad
    observada de cada período de accidente.

    Parámetros
    ----------
    triangulo : pd.DataFrame
        Triángulo acumulado en formato ancho.

    ldf_seleccionados : pd.Series o pd.DataFrame
        LDF de calcular_seleccion_ldf o seleccionar_ldf. Debe
        contener una transición por cada par de columnas consecutivas.

    primas : escalar, dict o pd.Series
        Prima ganada, exposición monetizada o base equivalente por
        período de accidente.

    ratio_perdida_esperada : escalar, dict o pd.Series
        Expected Loss Ratio (ELR) por período de accidente.

    factor_cola : float, opcional
        Factor desde el último desarrollo explícito hasta ultimate.

    validar_monotonia : bool, opcional
        Si es True, exige que el triángulo no disminuya entre
        desarrollos. Puede desactivarse para escenarios simulados.

    permitir_cdf_menor_uno : bool, opcional
        Si es True, permite CDF menores que 1 y resultados como
        reservas negativas para estudiar escenarios no normales.

    Retorna
    -------
    pd.DataFrame
        Resultado con observado, CDF a último, ultimate y reserva B-F.
    """

    # =========================================================
    # 1. Validaciones
    # =========================================================

    if not isinstance(triangulo, pd.DataFrame):
        raise TypeError("triangulo debe ser un pandas.DataFrame.")

    if triangulo.empty or triangulo.shape[1] < 2:
        raise ValueError(
            "triangulo debe tener al menos dos períodos de desarrollo."
        )

    if not isinstance(factor_cola, (int, float)):
        raise TypeError("factor_cola debe ser numérico.")

    if factor_cola <= 0:
        raise ValueError("factor_cola debe ser mayor que 0.")

    if not isinstance(validar_monotonia, bool):
        raise TypeError("validar_monotonia debe ser un booleano.")

    if not isinstance(permitir_cdf_menor_uno, bool):
        raise TypeError(
            "permitir_cdf_menor_uno debe ser un booleano."
        )

    if isinstance(ldf_seleccionados, pd.DataFrame):
        if ldf_seleccionados.shape[1] != 1:
            raise ValueError(
                "ldf_seleccionados debe tener una sola columna."
            )

        ldf = ldf_seleccionados.iloc[:, 0].copy()
    elif isinstance(ldf_seleccionados, pd.Series):
        ldf = ldf_seleccionados.copy()
    else:
        raise TypeError(
            "ldf_seleccionados debe ser un pandas.Series o DataFrame."
        )

    triangulo = validar_triangulo_acumulado(
        triangulo,
        exigir_no_decreciente=validar_monotonia,
    )

    if len(ldf) != triangulo.shape[1] - 1:
        raise ValueError(
            "ldf_seleccionados debe tener un LDF por cada transición "
            "del triángulo."
        )

    ldf = pd.to_numeric(ldf, errors="raise")

    if ldf.isna().any() or (ldf <= 0).any():
        raise ValueError(
            "Los LDF seleccionados deben ser valores positivos y no nulos."
        )

    primas_serie = _convertir_a_serie_bf(
        valor=primas,
        indice=triangulo.index,
        nombre="primas",
    )
    elr_serie = _convertir_a_serie_bf(
        valor=ratio_perdida_esperada,
        indice=triangulo.index,
        nombre="ratio_perdida_esperada",
    )

    # =========================================================
    # 2. Construir CDF a último a partir de los LDF
    # =========================================================

    cdf_por_edad = []

    for posicion in range(triangulo.shape[1]):
        cdf = ldf.iloc[posicion:].prod() * factor_cola
        cdf_por_edad.append(cdf)

    if not permitir_cdf_menor_uno and any(
        cdf < 1 for cdf in cdf_por_edad
    ):
        raise ValueError(
            "Los LDF y factor_cola deben producir CDF a último "
            "mayores o iguales a 1."
        )

    # =========================================================
    # 3. Calcular B-F por período de accidente
    # =========================================================

    resultado = []

    for periodo, fila in triangulo.iterrows():
        observados = fila.dropna()

        if observados.empty:
            raise ValueError(
                f"El período '{periodo}' no tiene valores observados."
            )

        posicion = triangulo.columns.get_loc(observados.index[-1])
        observado = observados.iloc[-1]
        cdf = cdf_por_edad[posicion]
        porcentaje_no_desarrollado = 1 - (1 / cdf)
        ultimate_esperado = (
            primas_serie.loc[periodo] * elr_serie.loc[periodo]
        )
        ultimate_bf = (
            observado
            + ultimate_esperado * porcentaje_no_desarrollado
        )

        resultado.append(
            {
                "periodo_accidente": periodo,
                "valor_observado": observado,
                "edad_desarrollo": observados.index[-1],
                "cdf_a_ultimo": cdf,
                "porcentaje_no_desarrollado": porcentaje_no_desarrollado,
                "prima": primas_serie.loc[periodo],
                "ratio_perdida_esperada": elr_serie.loc[periodo],
                "ultimate_esperado": ultimate_esperado,
                "ultimate_bf": ultimate_bf,
                "reserva_bf": ultimate_bf - observado,
            }
        )

    return pd.DataFrame(resultado).set_index("periodo_accidente")

def validar_triangulo_acumulado(
    triangulo: pd.DataFrame,
    exigir_no_decreciente: bool = True,
) -> pd.DataFrame:
    """
    Valida y ordena un triángulo acumulado en formato ancho.

    Por defecto, exige valores no decrecientes dentro de cada período
    de accidente. Esta condición se puede desactivar para estudiar
    escenarios acumulados con correcciones negativas simuladas.

    Parámetros
    ----------
    triangulo : pd.DataFrame
        Triángulo ancho a validar.

    exigir_no_decreciente : bool, opcional
        Si es True, rechaza disminuciones entre desarrollos.

    Retorna
    -------
    pd.DataFrame
        Copia numérica y ordenada del triángulo.
    """

    if not isinstance(triangulo, pd.DataFrame):
        raise TypeError(
            "triangulo debe ser un pandas.DataFrame."
        )

    if triangulo.empty:
        raise ValueError(
            "triangulo no puede estar vacío."
        )

    if not isinstance(exigir_no_decreciente, bool):
        raise TypeError(
            "exigir_no_decreciente debe ser un booleano."
        )

    resultado = triangulo.apply(
        pd.to_numeric,
        errors="raise",
    )

    resultado = (
        resultado.sort_index()
        .sort_index(axis=1)
    )

    if exigir_no_decreciente and (
        resultado.diff(axis=1) < 0
    ).any().any():
        raise ValueError(
            "triangulo debe contener valores acumulados no "
            "decrecientes. Si la base es incremental, usa "
            "acumular_triangulo antes de calcular reservas."
        )

    return resultado


def acumular_triangulo(
    triangulo: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convierte un triángulo incremental en acumulado.

    La acumulación se realiza horizontalmente por cada período de
    accidente, conservando la forma triangular y los valores NaN.

    Parámetros
    ----------
    triangulo : pd.DataFrame
        Triángulo incremental en formato ancho.

    Retorna
    -------
    pd.DataFrame
        Triángulo acumulado.
    """

    if not isinstance(triangulo, pd.DataFrame):
        raise TypeError(
            "triangulo debe ser un pandas.DataFrame."
        )

    if triangulo.empty:
        raise ValueError(
            "triangulo no puede estar vacío."
        )

    resultado = triangulo.apply(
        pd.to_numeric,
        errors="raise",
    )

    resultado = (
        resultado.sort_index()
        .sort_index(axis=1)
    )

    for periodo, fila in resultado.iterrows():
        observados = fila.notna().to_numpy()

        if observados.any():
            ultima_posicion = np.flatnonzero(observados)[-1]

            if not observados[:ultima_posicion + 1].all():
                raise ValueError(
                    f"El período '{periodo}' tiene huecos entre "
                    "desarrollos observados."
                )

    return resultado.cumsum(axis=1)


def _normalizar_ldf_reservas(
    ldf_seleccionados: pd.Series | pd.DataFrame,
    num_transiciones: int,
) -> pd.Series:
    """Valida y normaliza LDF para métodos de reservas."""

    if isinstance(ldf_seleccionados, pd.DataFrame):
        if ldf_seleccionados.shape[1] != 1:
            raise ValueError(
                "ldf_seleccionados debe tener una sola columna."
            )

        ldf = ldf_seleccionados.iloc[:, 0].copy()
    elif isinstance(ldf_seleccionados, pd.Series):
        ldf = ldf_seleccionados.copy()
    else:
        raise TypeError(
            "ldf_seleccionados debe ser un pandas.Series o DataFrame."
        )

    if len(ldf) != num_transiciones:
        raise ValueError(
            "ldf_seleccionados debe tener un LDF por cada transición "
            "del triángulo."
        )

    ldf = pd.to_numeric(
        ldf,
        errors="raise",
    )

    if ldf.isna().any() or (ldf <= 0).any():
        raise ValueError(
            "Los LDF seleccionados deben ser valores positivos y no nulos."
        )

    return ldf


def calcular_chain_ladder(
    triangulo: pd.DataFrame,
    ldf_seleccionados: pd.Series | pd.DataFrame,
    factor_cola: float = 1.0,
    validar_monotonia: bool = True,
    permitir_cdf_menor_uno: bool = False,
) -> pd.DataFrame:
    """
    Calcula ultimate y reserva mediante Chain Ladder.

    Para cada período de accidente, proyecta el último valor
    acumulado observado hasta último usando los LDF seleccionados:

        CDF_j = LDF_j x ... x LDF_ultimo x factor_cola
        Ultimate_CL = Observado_j x CDF_j
        Reserva_CL = Ultimate_CL - Observado_j

    Parámetros
    ----------
    triangulo : pd.DataFrame
        Triángulo acumulado en formato ancho.

    ldf_seleccionados : pd.Series o pd.DataFrame
        LDF de calcular_seleccion_ldf o seleccionar_ldf.

    factor_cola : float, opcional
        Factor desde el último desarrollo explícito hasta ultimate.

    validar_monotonia : bool, opcional
        Si es True, exige que el triángulo no disminuya entre
        desarrollos.

    permitir_cdf_menor_uno : bool, opcional
        Si es True, permite CDF menores que 1 para análisis de
        escenarios no normales.

    Retorna
    -------
    pd.DataFrame
        Resultado por período con observado, CDF, ultimate y reserva.
    """

    # =========================================================
    # 1. Validaciones
    # =========================================================

    if not isinstance(validar_monotonia, bool):
        raise TypeError("validar_monotonia debe ser un booleano.")

    if not isinstance(permitir_cdf_menor_uno, bool):
        raise TypeError(
            "permitir_cdf_menor_uno debe ser un booleano."
        )

    triangulo = validar_triangulo_acumulado(
        triangulo,
        exigir_no_decreciente=validar_monotonia,
    )

    if triangulo.shape[1] < 2:
        raise ValueError(
            "triangulo debe tener al menos dos períodos de desarrollo."
        )

    if not isinstance(factor_cola, (int, float)):
        raise TypeError(
            "factor_cola debe ser numérico."
        )

    if factor_cola <= 0:
        raise ValueError(
            "factor_cola debe ser mayor que 0."
        )

    ldf = _normalizar_ldf_reservas(
        ldf_seleccionados=ldf_seleccionados,
        num_transiciones=triangulo.shape[1] - 1,
    )

    # =========================================================
    # 2. Construir CDF a último
    # =========================================================

    cdf_por_edad = []

    for posicion in range(triangulo.shape[1]):
        cdf = ldf.iloc[posicion:].prod() * factor_cola
        cdf_por_edad.append(cdf)

    if not permitir_cdf_menor_uno and any(
        cdf < 1 for cdf in cdf_por_edad
    ):
        raise ValueError(
            "Los LDF y factor_cola deben producir CDF a último "
            "mayores o iguales a 1."
        )

    # =========================================================
    # 3. Calcular Chain Ladder
    # =========================================================

    resultado = []

    for periodo, fila in triangulo.iterrows():
        observados = fila.dropna()

        if observados.empty:
            raise ValueError(
                f"El período '{periodo}' no tiene valores observados."
            )

        posicion = triangulo.columns.get_loc(
            observados.index[-1]
        )
        observado = observados.iloc[-1]
        cdf = cdf_por_edad[posicion]
        ultimate = observado * cdf

        resultado.append(
            {
                "periodo_accidente": periodo,
                "valor_observado": observado,
                "edad_desarrollo": observados.index[-1],
                "cdf_a_ultimo": cdf,
                "ultimate_chain_ladder": ultimate,
                "reserva_chain_ladder": ultimate - observado,
            }
        )

    return pd.DataFrame(resultado).set_index(
        "periodo_accidente"
    )