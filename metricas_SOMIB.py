# metricas_articulo.py
# ------------------------------------------------------------
# Compara métricas de imágenes originales vs imágenes procesadas
# usando el pipeline de Metodo_SOMIB.py
# ------------------------------------------------------------

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from Metodo_SOMIB import analizar_imagen, decidir_pipeline


# ============================================================
# 1. MÉTRICAS DE UNA IMAGEN
# ============================================================
def tabla_comparacion_pre_post(resumen):
    """
    Genera una tabla limpia para artículo:
    Métrica | Original media ± DE | Procesada media ± DE | Δ medio | Cambio %
    """

    tabla = pd.DataFrame(
        {
            "Métrica": resumen["metrica"],
            "Original (media ± DE)": resumen.apply(
                lambda r: f"{r['pre_media']:.3f} ± {r['pre_std']:.3f}", axis=1
            ),
            "Procesada (media ± DE)": resumen.apply(
                lambda r: f"{r['post_media']:.3f} ± {r['post_std']:.3f}", axis=1
            ),
            "Δ medio": resumen["delta_media"].round(3),
            "Cambio medio (%)": resumen["cambio_pct_medio"].round(2),
        }
    )

    return tabla


def tabla_distribucion_pipeline(df):
    """
    Genera tabla de distribución de pipelines aplicados.
    """

    conteo = (
        df["pasos"]
        .fillna("ninguno")
        .replace("", "ninguno")
        .value_counts()
        .reset_index()
    )

    conteo.columns = ["Pipeline aplicado", "Cantidad"]
    conteo["Porcentaje (%)"] = 100 * conteo["Cantidad"] / len(df)
    conteo["Porcentaje (%)"] = conteo["Porcentaje (%)"].round(2)

    return conteo


def metricas_imagen(img):
    """
    Calcula métricas globales de calidad para una imagen en escala de grises.

    Métricas:
    - rango_usado
    - media
    - std
    - rango_iluminacion
    - sigma_ruido
    - nitidez
    - gradiente_medio
    - entropia
    """
    if img is None:
        raise ValueError("La imagen recibida es None.")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.uint8)

    H, W = img.shape

    # Iluminación de fondo: componente de baja frecuencia
    sigma_ilum = max(W, H) // 8
    sigma_ilum = max(1, sigma_ilum)

    iluminacion = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma_ilum)
    rango_iluminacion = float(iluminacion.max() - iluminacion.min())

    # Ruido estimado: residuo contra filtro de mediana
    img_suave = cv2.medianBlur(img, 5)
    ruido = img.astype(np.float32) - img_suave.astype(np.float32)
    sigma_ruido = float(ruido.std())

    # Nitidez: varianza del Laplaciano
    lap = cv2.Laplacian(img, cv2.CV_32F)
    nitidez = float(lap.var())

    # Gradiente medio
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    gradiente_medio = float(np.sqrt(gx**2 + gy**2).mean())

    # Entropía global
    hist = np.bincount(img.flatten(), minlength=256).astype(np.float32)
    p = hist / (hist.sum() + 1e-8)
    p = p[p > 0]
    entropia = float(-np.sum(p * np.log2(p + 1e-8)))

    return {
        "alto": H,
        "ancho": W,
        "intensidad_min": int(img.min()),
        "intensidad_max": int(img.max()),
        "rango_usado": int(img.max() - img.min()),
        "media": float(img.mean()),
        "std": float(img.std()),
        "rango_iluminacion": rango_iluminacion,
        "sigma_ruido": sigma_ruido,
        "nitidez": nitidez,
        "gradiente_medio": gradiente_medio,
        "entropia": entropia,
    }


# ============================================================
# 2. COMPARACIÓN ORIGINAL VS PROCESADA
# ============================================================


def comparar_original_vs_procesada(
    carpeta,
    salida_imagenes=None,
    salida_csv="metricas_pre_post.csv",
    salida_resumen_csv="resumen_articulo.csv",
    extensiones=(".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"),
    umbrales=None,
):
    """
    Recorre una carpeta de imágenes, aplica el pipeline de preprocesamiento
    y calcula métricas antes y después.

    Parámetros
    ----------
    carpeta : str or Path
        Carpeta con las imágenes originales.

    salida_imagenes : str or Path or None
        Carpeta donde se guardarán las imágenes procesadas.
        Si es None, no guarda imágenes procesadas.

    salida_csv : str
        Archivo CSV con métricas por imagen.

    salida_resumen_csv : str
        Archivo CSV con resumen para artículo.

    extensiones : tuple
        Extensiones de imagen permitidas.

    umbrales : dict or None
        Umbrales personalizados para decidir_pipeline.
        Si es None, usa los umbrales por defecto de Metodo_SOMIB.py.

    Retorna
    -------
    df : pandas.DataFrame
        Métricas por imagen.

    resumen : pandas.DataFrame
        Resumen estadístico para artículo.
    """

    carpeta = Path(carpeta)

    if not carpeta.exists():
        raise FileNotFoundError(f"No existe la carpeta: {carpeta}")

    if salida_imagenes is not None:
        salida_imagenes = Path(salida_imagenes)
        salida_imagenes.mkdir(parents=True, exist_ok=True)

    archivos = sorted(f for f in carpeta.rglob("*") if f.suffix.lower() in extensiones)

    if len(archivos) == 0:
        raise FileNotFoundError(f"No se encontraron imágenes en: {carpeta}")

    print(f"Imágenes encontradas: {len(archivos)}")

    filas = []

    for archivo in tqdm(archivos, desc="Calculando métricas pre/post"):
        img = cv2.imread(str(archivo), cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[ADVERTENCIA] No se pudo leer: {archivo}")
            continue

        # -----------------------------
        # Métricas originales
        # -----------------------------
        met_pre = metricas_imagen(img)

        # -----------------------------
        # Procesamiento
        # -----------------------------
        diag = analizar_imagen(img)

        if umbrales is None:
            img_proc, diag, _ = decidir_pipeline(
                diag,
                img,
                guardar_historial=False,
            )
        else:
            img_proc, diag, _ = decidir_pipeline(
                diag,
                img,
                umbrales=umbrales,
                guardar_historial=False,
            )

        img_proc = img_proc.astype(np.uint8)

        # Guardar imagen procesada
        if salida_imagenes is not None:
            cv2.imwrite(str(salida_imagenes / archivo.name), img_proc)

        # -----------------------------
        # Métricas procesadas
        # -----------------------------
        met_post = metricas_imagen(img_proc)

        fila = {
            "archivo": archivo.name,
            "ruta": str(archivo),
            "alto": met_pre["alto"],
            "ancho": met_pre["ancho"],
            "n_pasos_aplicados": len(diag.pasos_recomendados),
            "pasos": "→".join(p["nombre"] for p in diag.pasos_recomendados),
        }

        # Columnas pre/post/delta/cambio %
        for k in met_pre.keys():
            if k in ["alto", "ancho"]:
                continue

            pre = met_pre[k]
            post = met_post[k]

            fila[f"{k}_pre"] = pre
            fila[f"{k}_post"] = post
            fila[f"{k}_delta"] = post - pre
            fila[f"{k}_cambio_pct"] = 100.0 * (post - pre) / (abs(pre) + 1e-8)

        filas.append(fila)

    df = pd.DataFrame(filas)

    if len(df) == 0:
        raise RuntimeError("No se pudo calcular ninguna métrica. Revisa la carpeta.")

    # Guardar CSV completo
    if salida_csv is not None:
        df.to_csv(salida_csv, index=False)
        print(f"CSV completo guardado en: {salida_csv}")

    # Crear resumen para artículo
    resumen = crear_resumen_articulo(df)

    if salida_resumen_csv is not None:
        resumen.to_csv(salida_resumen_csv, index=False)
        print(f"Resumen guardado en: {salida_resumen_csv}")

    return df, resumen


# ============================================================
# 3. RESUMEN PARA ARTÍCULO
# ============================================================


def crear_resumen_articulo(df):
    """
    Genera una tabla resumen con media, desviación estándar,
    mediana, delta y cambio porcentual.

    Esta tabla es la que conviene reportar en el artículo.
    """

    metricas_articulo = [
        "rango_usado",
        "std",
        "rango_iluminacion",
        "sigma_ruido",
        "nitidez",
        "gradiente_medio",
        "entropia",
    ]

    resumen_filas = []

    for m in metricas_articulo:
        pre = df[f"{m}_pre"]
        post = df[f"{m}_post"]
        delta = df[f"{m}_delta"]
        pct = df[f"{m}_cambio_pct"]

        resumen_filas.append(
            {
                "metrica": m,
                "pre_media": pre.mean(),
                "pre_std": pre.std(),
                "pre_mediana": pre.median(),
                "post_media": post.mean(),
                "post_std": post.std(),
                "post_mediana": post.median(),
                "delta_media": delta.mean(),
                "delta_mediana": delta.median(),
                "cambio_pct_medio": pct.mean(),
                "cambio_pct_mediana": pct.median(),
            }
        )

    resumen = pd.DataFrame(resumen_filas)

    return resumen


def imprimir_resumen_articulo(resumen):
    """
    Imprime el resumen en formato legible para revisar rápido.
    """

    print("\n" + "=" * 100)
    print("RESUMEN PARA ARTÍCULO")
    print("=" * 100)

    columnas = [
        "metrica",
        "pre_media",
        "pre_std",
        "post_media",
        "post_std",
        "delta_media",
        "cambio_pct_medio",
    ]

    print(resumen[columnas].round(3).to_string(index=False))
    print("=" * 100)


# ============================================================
# 4. GRÁFICAS
# ============================================================


def graficar_boxplots_pre_post(
    df,
    carpeta_salida="figuras_metricas",
    mostrar=True,
):
    """
    Genera boxplots comparando original vs procesada para cada métrica.
    Guarda cada figura como PNG.
    """

    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    metricas = [
        "rango_usado",
        "std",
        "rango_iluminacion",
        "sigma_ruido",
        "nitidez",
        "gradiente_medio",
        "entropia",
    ]

    for m in metricas:
        datos = [
            df[f"{m}_pre"].dropna(),
            df[f"{m}_post"].dropna(),
        ]

        plt.figure(figsize=(5, 4))
        plt.boxplot(datos, labels=["Original", "Procesada"])
        plt.title(m)
        plt.ylabel("Valor")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        ruta_fig = carpeta_salida / f"boxplot_{m}.png"
        plt.savefig(ruta_fig, dpi=300)

        if mostrar:
            plt.show()
        else:
            plt.close()

    print(f"Boxplots guardados en: {carpeta_salida}")


def graficar_histogramas_pre_post(
    df,
    carpeta_salida="figuras_metricas",
    mostrar=True,
):
    """
    Genera histogramas superpuestos original vs procesada.
    Guarda cada figura como PNG.
    """

    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    metricas = [
        "rango_usado",
        "std",
        "rango_iluminacion",
        "sigma_ruido",
        "nitidez",
        "gradiente_medio",
        "entropia",
    ]

    for m in metricas:
        pre = df[f"{m}_pre"].dropna()
        post = df[f"{m}_post"].dropna()

        vmin = min(pre.min(), post.min())
        vmax = max(pre.max(), post.max())

        if np.isclose(vmin, vmax):
            bins = 10
        else:
            bins = np.linspace(vmin, vmax, 30)

        plt.figure(figsize=(6, 4))
        plt.hist(pre, bins=bins, alpha=0.55, label="Original", edgecolor="white")
        plt.hist(post, bins=bins, alpha=0.55, label="Procesada", edgecolor="white")
        plt.title(m)
        plt.xlabel("Valor")
        plt.ylabel("Frecuencia")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        ruta_fig = carpeta_salida / f"histograma_{m}.png"
        plt.savefig(ruta_fig, dpi=300)

        if mostrar:
            plt.show()
        else:
            plt.close()

    print(f"Histogramas guardados en: {carpeta_salida}")


def graficar_barras_cambio_pct(
    resumen,
    carpeta_salida="figuras_metricas",
    mostrar=True,
):
    """
    Grafica el cambio porcentual medio de cada métrica.
    """

    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 4))
    plt.bar(resumen["metrica"], resumen["cambio_pct_medio"])
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Cambio porcentual medio (%)")
    plt.title("Cambio porcentual medio después del procesamiento")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    ruta_fig = carpeta_salida / "cambio_pct_medio.png"
    plt.savefig(ruta_fig, dpi=300)

    if mostrar:
        plt.show()
    else:
        plt.close()

    print(f"Gráfica de cambio porcentual guardada en: {ruta_fig}")


# ============================================================
# 5. ESTADÍSTICAS DEL PIPELINE
# ============================================================


def estadisticas_pipeline(df):
    """
    Cuenta cuántas imágenes recibieron cada combinación de pasos.
    """

    print("\n" + "=" * 90)
    print("DISTRIBUCIÓN DE PIPELINES APLICADOS")
    print("=" * 90)

    conteo = df["pasos"].fillna("ninguno").replace("", "ninguno").value_counts()
    total = len(df)

    print(f"{'Pipeline':<60} {'Cantidad':>10} {'%':>8}")
    print("-" * 90)

    for combinacion, n in conteo.items():
        print(f"{combinacion:<60} {n:>10} {n / total * 100:>7.1f}%")

    print("=" * 90)
    print(f"Número medio de pasos aplicados: {df['n_pasos_aplicados'].mean():.2f}")
    print("\nHistograma de pasos aplicados:")
    print(df["n_pasos_aplicados"].value_counts().sort_index())


# ============================================================
# 6. VARIACION E IMPORTANCIA DE FEATURES DE REGIONES
# ============================================================


def _normalizar_columna_importancia(valores):
    valores = pd.Series(valores, dtype=float).replace([np.inf, -np.inf], np.nan)
    valores = valores.fillna(0.0)
    vmin = float(valores.min())
    vmax = float(valores.max())

    if np.isclose(vmin, vmax):
        return pd.Series(np.zeros(len(valores)), index=valores.index, dtype=float)

    return (valores - vmin) / (vmax - vmin)


def construir_dataframe_features_regiones(ruta_etiquetas, carpeta_imagenes):
    """
    Reconstruye la matriz de las 30 features usadas por etiquetar_regiones_carpeta.

    Parametros
    ----------
    ruta_etiquetas : str or Path
        Pickle generado por etiquetar_regiones_carpeta.

    carpeta_imagenes : str or Path
        Carpeta con las imagenes originales correspondientes a las etiquetas.

    Retorna
    -------
    pandas.DataFrame
        Una fila por region etiquetada con columnas de metadatos y features.
    """
    from etiquetar_regiones_carpeta import (
        FEATURES_OBJETO,
        _extraer_features_objeto,
        cargar_etiquetas,
    )

    ruta_etiquetas = Path(ruta_etiquetas).expanduser()
    carpeta_imagenes = Path(carpeta_imagenes).expanduser()

    if not ruta_etiquetas.exists():
        raise FileNotFoundError(f"No existe el archivo de etiquetas: {ruta_etiquetas}")

    if not carpeta_imagenes.exists():
        raise FileNotFoundError(f"No existe la carpeta de imagenes: {carpeta_imagenes}")

    datos = cargar_etiquetas(str(ruta_etiquetas))
    feature_names = list(FEATURES_OBJETO)
    filas = []
    omitidas = []

    for nombre, d in datos.items():
        etiquetas = d.get("etiquetas", {})
        if not etiquetas:
            continue

        labels_map = d.get("labels_map")
        if labels_map is None:
            omitidas.append((nombre, "sin labels_map"))
            continue

        ruta_img = carpeta_imagenes / nombre
        if not ruta_img.exists():
            coincidencias = list(carpeta_imagenes.rglob(nombre))
            ruta_img = coincidencias[0] if coincidencias else ruta_img

        img_bgr = cv2.imread(str(ruta_img))
        if img_bgr is None:
            omitidas.append((nombre, "imagen no encontrada o ilegible"))
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        for lab, cls in etiquetas.items():
            feat = np.asarray(
                _extraer_features_objeto(img_rgb, labels_map, int(lab)),
                dtype=np.float32,
            ).reshape(-1)

            if len(feat) != len(feature_names):
                raise ValueError(
                    f"La region {nombre}:{lab} genero {len(feat)} features, "
                    f"pero FEATURES_OBJETO define {len(feature_names)}."
                )

            fila = {col: float(val) for col, val in zip(feature_names, feat)}
            fila.update(
                {
                    "archivo": nombre,
                    "lab": int(lab),
                    "clase": int(cls),
                    "clase_nombre": "ameba" if int(cls) == 1 else "no_ameba",
                }
            )
            filas.append(fila)

    if not filas:
        raise ValueError(
            "No se pudo construir el dataset de features. Revisa que el pickle "
            "tenga regiones etiquetadas y que las imagenes existan en la carpeta dada."
        )

    if omitidas:
        print(f"[ADVERTENCIA] Se omitieron {len(omitidas)} imagenes con problema.")
        for nombre, razon in omitidas[:10]:
            print(f"  - {nombre}: {razon}")
        if len(omitidas) > 10:
            print(f"  ... y {len(omitidas) - 10} mas")

    df = pd.DataFrame(filas)
    columnas = ["archivo", "lab", "clase", "clase_nombre"] + feature_names

    return df[columnas]


def tabla_variacion_features_regiones(df_features, feature_names=None):
    """
    Resume variabilidad global y separacion entre clases por feature.
    """
    if feature_names is None:
        feature_names = [
            c
            for c in df_features.columns
            if c not in {"archivo", "lab", "clase", "clase_nombre"}
        ]

    y = df_features["clase"].astype(int)
    filas = []

    for feature in feature_names:
        x = pd.to_numeric(df_features[feature], errors="coerce")
        x0 = x[y == 0].dropna()
        x1 = x[y == 1].dropna()
        x_all = x.dropna()

        media = float(x_all.mean())
        std = float(x_all.std(ddof=1)) if len(x_all) > 1 else 0.0
        varianza = float(x_all.var(ddof=1)) if len(x_all) > 1 else 0.0
        q1 = float(x_all.quantile(0.25))
        q3 = float(x_all.quantile(0.75))
        iqr = q3 - q1
        cv = std / (abs(media) + 1e-8)

        media_0 = float(x0.mean()) if len(x0) else np.nan
        media_1 = float(x1.mean()) if len(x1) else np.nan
        std_0 = float(x0.std(ddof=1)) if len(x0) > 1 else 0.0
        std_1 = float(x1.std(ddof=1)) if len(x1) > 1 else 0.0
        delta_medias = media_1 - media_0

        n0 = len(x0)
        n1 = len(x1)
        if n0 + n1 > 2:
            pooled_var = ((n0 - 1) * std_0**2 + (n1 - 1) * std_1**2) / (n0 + n1 - 2)
            pooled_std = float(np.sqrt(max(pooled_var, 0.0)))
        else:
            pooled_std = 0.0

        cohen_d = delta_medias / (pooled_std + 1e-8)

        filas.append(
            {
                "feature": feature,
                "n": int(len(x_all)),
                "media": media,
                "std": std,
                "varianza": varianza,
                "cv": cv,
                "min": float(x_all.min()),
                "q1": q1,
                "mediana": float(x_all.median()),
                "q3": q3,
                "max": float(x_all.max()),
                "iqr": iqr,
                "media_no_ameba": media_0,
                "media_ameba": media_1,
                "delta_medias_ameba_menos_no_ameba": delta_medias,
                "cohen_d": cohen_d,
                "abs_cohen_d": abs(cohen_d),
            }
        )

    return pd.DataFrame(filas).sort_values("abs_cohen_d", ascending=False)


def tabla_importancia_features_regiones(
    df_features,
    feature_names=None,
    random_state=42,
    n_estimators=500,
    test_size=0.30,
):
    """
    Calcula importancia supervisada de features con las etiquetas manuales.

    Combina:
    - importancia Gini de Random Forest
    - importancia por permutacion si el split estratificado es posible
    - informacion mutua
    - tamano de efecto entre clases (abs_cohen_d)
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import train_test_split

    if feature_names is None:
        feature_names = [
            c
            for c in df_features.columns
            if c not in {"archivo", "lab", "clase", "clase_nombre"}
        ]

    X = df_features[feature_names].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    y = df_features["clase"].astype(int).to_numpy()

    clases, conteos = np.unique(y, return_counts=True)
    if len(clases) < 2:
        raise ValueError(
            "La importancia supervisada necesita etiquetas de ambas clases: "
            "ameba (1) y no_ameba (0)."
        )

    split_valido = False
    try:
        Xtr, Xval, ytr, yval = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        split_valido = True
    except ValueError:
        Xtr, Xval, ytr, yval = X, X, y, y

    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    modelo.fit(Xtr, ytr)

    pred = modelo.predict(Xval)
    metricas_modelo = pd.DataFrame(
        [
            {
                "n_muestras": int(len(y)),
                "n_no_ameba": int(np.sum(y == 0)),
                "n_ameba": int(np.sum(y == 1)),
                "split_validacion": bool(split_valido),
                "accuracy": float(accuracy_score(yval, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(yval, pred)),
            }
        ]
    )

    if split_valido:
        perm = permutation_importance(
            modelo,
            Xval,
            yval,
            scoring="balanced_accuracy",
            n_repeats=30,
            random_state=random_state,
            n_jobs=-1,
        )
        importancia_perm_media = perm.importances_mean
        importancia_perm_std = perm.importances_std
    else:
        importancia_perm_media = np.full(len(feature_names), np.nan)
        importancia_perm_std = np.full(len(feature_names), np.nan)

    try:
        mutual_info = mutual_info_classif(X, y, random_state=random_state)
    except ValueError:
        mutual_info = np.zeros(len(feature_names), dtype=float)

    variacion = tabla_variacion_features_regiones(df_features, feature_names)
    tabla = pd.DataFrame(
        {
            "feature": feature_names,
            "importancia_rf": modelo.feature_importances_,
            "importancia_perm_media": importancia_perm_media,
            "importancia_perm_std": importancia_perm_std,
            "mutual_info": mutual_info,
        }
    ).merge(
        variacion[
            [
                "feature",
                "cv",
                "iqr",
                "delta_medias_ameba_menos_no_ameba",
                "cohen_d",
                "abs_cohen_d",
            ]
        ],
        on="feature",
        how="left",
    )

    columnas_score = [
        _normalizar_columna_importancia(tabla["importancia_rf"]),
        _normalizar_columna_importancia(tabla["mutual_info"]),
        _normalizar_columna_importancia(tabla["abs_cohen_d"]),
        _normalizar_columna_importancia(
            pd.Series(tabla["importancia_perm_media"]).clip(lower=0.0)
        ),
    ]
    tabla["score_importancia"] = pd.concat(columnas_score, axis=1).mean(axis=1)

    tabla = tabla.sort_values("score_importancia", ascending=False).reset_index(
        drop=True
    )

    return tabla, metricas_modelo


def graficar_importancia_features_regiones(
    tabla_importancia,
    carpeta_salida="figuras_metricas",
    top_n=15,
    mostrar=True,
):
    """
    Grafica el ranking de importancia combinado para las features de regiones.
    """
    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    top = tabla_importancia.head(top_n).iloc[::-1]

    plt.figure(figsize=(9, max(4, 0.35 * len(top))))
    plt.barh(top["feature"], top["score_importancia"])
    plt.xlabel("Score combinado de importancia")
    plt.title("Importancia de features de regiones etiquetadas")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    ruta_fig = carpeta_salida / "importancia_features_regiones.png"
    plt.savefig(ruta_fig, dpi=300)

    if mostrar:
        plt.show()
    else:
        plt.close()

    print(f"Grafica de importancia guardada en: {ruta_fig}")


def analizar_variacion_importancia_features_regiones(
    ruta_etiquetas="../etiquetas_regiones_SOMIB.pkl",
    carpeta_imagenes="full_dataset",
    carpeta_salida="tablas_features",
    carpeta_figuras="figuras_metricas",
    guardar_csv=True,
    guardar_figura=True,
    mostrar=True,
    top_n=15,
    random_state=42,
):
    """
    Pipeline completo para medir variacion e importancia de las 30 features.

    Con las etiquetas es suficiente si:
    - el pickle conserva labels_map y etiquetas;
    - hay ejemplos de ambas clases;
    - carpeta_imagenes contiene las imagenes originales con los mismos nombres.
    """
    df_features = construir_dataframe_features_regiones(
        ruta_etiquetas=ruta_etiquetas,
        carpeta_imagenes=carpeta_imagenes,
    )

    feature_names = [
        c
        for c in df_features.columns
        if c not in {"archivo", "lab", "clase", "clase_nombre"}
    ]

    print("\n" + "=" * 90)
    print("ANALISIS DE FEATURES DE REGIONES")
    print("=" * 90)
    print(f"Features detectadas : {len(feature_names)}")
    print(f"Regiones etiquetadas: {len(df_features)}")
    print(df_features["clase_nombre"].value_counts().to_string())

    variacion = tabla_variacion_features_regiones(df_features, feature_names)
    importancia, metricas_modelo = tabla_importancia_features_regiones(
        df_features,
        feature_names=feature_names,
        random_state=random_state,
    )

    if guardar_csv:
        carpeta_salida = Path(carpeta_salida)
        carpeta_salida.mkdir(parents=True, exist_ok=True)

        df_features.to_csv(
            carpeta_salida / "features_regiones_etiquetadas.csv", index=False
        )
        variacion.to_csv(
            carpeta_salida / "tabla_15_variacion_features_regiones.csv",
            index=False,
        )
        importancia.to_csv(
            carpeta_salida / "tabla_16_importancia_features_regiones.csv",
            index=False,
        )
        metricas_modelo.to_csv(
            carpeta_salida / "tabla_17_metricas_modelo_importancia.csv",
            index=False,
        )

        print(f"Tablas de features guardadas en: {carpeta_salida}")

    if guardar_figura:
        graficar_importancia_features_regiones(
            importancia,
            carpeta_salida=carpeta_figuras,
            top_n=top_n,
            mostrar=mostrar,
        )

    print("\nTop features por score combinado:")
    print(
        importancia[
            [
                "feature",
                "score_importancia",
                "importancia_rf",
                "mutual_info",
                "abs_cohen_d",
            ]
        ]
        .head(top_n)
        .round(4)
        .to_string(index=False)
    )
    print("=" * 90)

    return {
        "features": df_features,
        "variacion": variacion,
        "importancia": importancia,
        "metricas_modelo": metricas_modelo,
    }


import numpy as np
import pandas as pd

METRICAS_ARTICULO = [
    "rango_usado",
    "std",
    "rango_iluminacion",
    "sigma_ruido",
    "nitidez",
    "gradiente_medio",
    "entropia",
]


# Dirección esperada de mejora:
# +1 = conviene que aumente
# -1 = conviene que disminuya
DIRECCION_MEJORA = {
    "rango_usado": +1,
    "std": +1,
    "rango_iluminacion": -1,
    "sigma_ruido": -1,
    "nitidez": +1,
    "gradiente_medio": +1,
    "entropia": +1,
}


def tabla_1_comparacion_pre_post(resumen):
    """
    Tabla principal:
    Métrica | Original media ± DE | Procesada media ± DE | Delta | Cambio %
    """
    tabla = pd.DataFrame(
        {
            "Métrica": resumen["metrica"],
            "Original (media ± DE)": resumen.apply(
                lambda r: f"{r['pre_media']:.3f} ± {r['pre_std']:.3f}", axis=1
            ),
            "Procesada (media ± DE)": resumen.apply(
                lambda r: f"{r['post_media']:.3f} ± {r['post_std']:.3f}", axis=1
            ),
            "Δ medio": resumen["delta_media"].round(3),
            "Cambio medio (%)": resumen["cambio_pct_medio"].round(2),
        }
    )

    return tabla


def tabla_2_descriptiva_original(df, metricas=METRICAS_ARTICULO):
    """
    Describe únicamente las imágenes originales.
    Buena para caracterizar el dataset de entrada.
    """
    filas = []

    for m in metricas:
        x = df[f"{m}_pre"].dropna()

        filas.append(
            {
                "Métrica": m,
                "Media": x.mean(),
                "DE": x.std(),
                "Mínimo": x.min(),
                "Q1": x.quantile(0.25),
                "Mediana": x.median(),
                "Q3": x.quantile(0.75),
                "Máximo": x.max(),
            }
        )

    return pd.DataFrame(filas).round(3)


def tabla_3_descriptiva_procesada(df, metricas=METRICAS_ARTICULO):
    """
    Describe únicamente las imágenes procesadas.
    Buena para reportar el estado final del dataset.
    """
    filas = []

    for m in metricas:
        x = df[f"{m}_post"].dropna()

        filas.append(
            {
                "Métrica": m,
                "Media": x.mean(),
                "DE": x.std(),
                "Mínimo": x.min(),
                "Q1": x.quantile(0.25),
                "Mediana": x.median(),
                "Q3": x.quantile(0.75),
                "Máximo": x.max(),
            }
        )

    return pd.DataFrame(filas).round(3)


def tabla_4_robusta_mediana_iqr(df, metricas=METRICAS_ARTICULO):
    """
    Tabla robusta:
    Mediana e IQR antes/después.
    Útil si hay muchos outliers.
    """
    filas = []

    for m in metricas:
        pre = df[f"{m}_pre"].dropna()
        post = df[f"{m}_post"].dropna()

        pre_iqr = pre.quantile(0.75) - pre.quantile(0.25)
        post_iqr = post.quantile(0.75) - post.quantile(0.25)

        filas.append(
            {
                "Métrica": m,
                "Original mediana": pre.median(),
                "Original IQR": pre_iqr,
                "Procesada mediana": post.median(),
                "Procesada IQR": post_iqr,
                "Δ mediana": post.median() - pre.median(),
                "Cambio IQR (%)": 100 * (post_iqr - pre_iqr) / (abs(pre_iqr) + 1e-8),
            }
        )

    return pd.DataFrame(filas).round(3)


def tabla_5_coeficiente_variacion(df, metricas=METRICAS_ARTICULO):
    """
    Compara la variabilidad relativa del dataset.
    CV = DE / media.
    Menor CV suele indicar mayor homogeneidad.
    """
    filas = []

    for m in metricas:
        pre = df[f"{m}_pre"].dropna()
        post = df[f"{m}_post"].dropna()

        cv_pre = pre.std() / (abs(pre.mean()) + 1e-8)
        cv_post = post.std() / (abs(post.mean()) + 1e-8)

        filas.append(
            {
                "Métrica": m,
                "CV original": cv_pre,
                "CV procesada": cv_post,
                "Δ CV": cv_post - cv_pre,
                "Cambio CV (%)": 100 * (cv_post - cv_pre) / (abs(cv_pre) + 1e-8),
            }
        )

    return pd.DataFrame(filas).round(4)


def tabla_6_porcentaje_imagenes_mejoradas(df, metricas=METRICAS_ARTICULO):
    """
    Cuenta en cuántas imágenes cada métrica mejoró según la dirección esperada.
    """
    filas = []
    n = len(df)

    for m in metricas:
        delta = df[f"{m}_delta"]

        direccion = DIRECCION_MEJORA.get(m, +1)

        if direccion == +1:
            mejoro = delta > 0
            empeoro = delta < 0
        else:
            mejoro = delta < 0
            empeoro = delta > 0

        filas.append(
            {
                "Métrica": m,
                "Imágenes mejoradas": int(mejoro.sum()),
                "Imágenes empeoradas": int(empeoro.sum()),
                "Sin cambio": int((delta == 0).sum()),
                "% mejoradas": 100 * mejoro.sum() / n,
                "% empeoradas": 100 * empeoro.sum() / n,
            }
        )

    return pd.DataFrame(filas).round(2)


def tabla_7_distribucion_pipeline(df):
    """
    Distribución de combinaciones de pasos aplicados.
    """
    conteo = (
        df["pasos"]
        .fillna("ninguno")
        .replace("", "ninguno")
        .value_counts()
        .reset_index()
    )

    conteo.columns = ["Pipeline aplicado", "Cantidad"]
    conteo["Porcentaje (%)"] = 100 * conteo["Cantidad"] / len(df)

    return conteo.round(2)


def tabla_8_numero_pasos(df):
    """
    Distribución del número de pasos aplicados.
    """
    tabla = df["n_pasos_aplicados"].value_counts().sort_index().reset_index()

    tabla.columns = ["Número de pasos", "Cantidad"]
    tabla["Porcentaje (%)"] = 100 * tabla["Cantidad"] / len(df)

    return tabla.round(2)


def tabla_9_condiciones_originales(
    df,
    umbrales=None,
):
    """
    Tabla de problemas detectados en las imágenes originales.
    Usa las métricas *_pre.
    """

    if umbrales is None:
        umbrales = {
            "rango_iluminacion_alto": 61.03,
            "sigma_ruido_alto": 11.46,
            "sigma_ruido_muy_alto": 8.09,
            "nitidez_baja": 201.22,
            "nitidez_muy_baja": 190.51,
            "std_baja": 5.01,
            "rango_comprimido": 168.97,
        }

    condiciones = {
        "Iluminación desigual": df["rango_iluminacion_pre"]
        > umbrales["rango_iluminacion_alto"],
        "Ruido alto": df["sigma_ruido_pre"] > umbrales["sigma_ruido_alto"],
        "Ruido muy alto": df["sigma_ruido_pre"] > umbrales["sigma_ruido_muy_alto"],
        "Nitidez baja": df["nitidez_pre"] < umbrales["nitidez_baja"],
        "Nitidez muy baja": df["nitidez_pre"] < umbrales["nitidez_muy_baja"],
        "Contraste pobre": df["std_pre"] < umbrales["std_baja"],
        "Rango comprimido": df["rango_usado_pre"] < umbrales["rango_comprimido"],
    }

    filas = []
    n = len(df)

    for nombre, mask in condiciones.items():
        filas.append(
            {
                "Condición detectada": nombre,
                "Cantidad": int(mask.sum()),
                "Porcentaje (%)": 100 * mask.sum() / n,
            }
        )

    return pd.DataFrame(filas).round(2)


def tabla_10_ranking_cambios(resumen):
    """
    Ordena las métricas por magnitud de cambio porcentual medio.
    """
    tabla = resumen[
        [
            "metrica",
            "delta_media",
            "cambio_pct_medio",
            "cambio_pct_mediana",
        ]
    ].copy()

    tabla["Magnitud cambio (%)"] = tabla["cambio_pct_medio"].abs()
    tabla = tabla.sort_values("Magnitud cambio (%)", ascending=False)

    return tabla.round(3)


def tabla_11_correlacion_metricas_cambios(df, metricas=METRICAS_ARTICULO):
    """
    Correlación entre el valor original de una métrica y su cambio.
    Sirve para discutir si el pipeline ayuda más a imágenes más degradadas.
    """
    filas = []

    for m in metricas:
        x = df[f"{m}_pre"]
        y = df[f"{m}_delta"]

        corr = x.corr(y)

        filas.append(
            {
                "Métrica": m,
                "Correlación original vs Δ": corr,
            }
        )

    return pd.DataFrame(filas).round(4)


def tabla_12_extremos_por_metrica(df, n=5, metricas=METRICAS_ARTICULO):
    """
    Devuelve los casos extremos por métrica:
    - mayores valores originales
    - menores valores originales
    - mayor cambio positivo
    - mayor cambio negativo
    """
    filas = []

    for m in metricas:
        col_pre = f"{m}_pre"
        col_delta = f"{m}_delta"

        top_altos = df.nlargest(n, col_pre)
        top_bajos = df.nsmallest(n, col_pre)
        top_delta_pos = df.nlargest(n, col_delta)
        top_delta_neg = df.nsmallest(n, col_delta)

        for tipo, subdf, col in [
            ("Original más alto", top_altos, col_pre),
            ("Original más bajo", top_bajos, col_pre),
            ("Mayor aumento", top_delta_pos, col_delta),
            ("Mayor disminución", top_delta_neg, col_delta),
        ]:
            for _, r in subdf.iterrows():
                filas.append(
                    {
                        "Métrica": m,
                        "Tipo extremo": tipo,
                        "Archivo": r["archivo"],
                        "Valor": r[col],
                    }
                )

    return pd.DataFrame(filas).round(3)


def tabla_13_resumen_por_pipeline(df, metricas=METRICAS_ARTICULO):
    """
    Resume el cambio promedio por cada pipeline aplicado.
    Útil para saber qué combinación de pasos tuvo mayor efecto.
    """
    df_tmp = df.copy()
    df_tmp["pasos"] = df_tmp["pasos"].fillna("ninguno").replace("", "ninguno")

    filas = []

    for pipeline, grupo in df_tmp.groupby("pasos"):
        fila = {
            "Pipeline": pipeline,
            "Cantidad": len(grupo),
        }

        for m in metricas:
            fila[f"{m}_delta_medio"] = grupo[f"{m}_delta"].mean()
            fila[f"{m}_cambio_pct_medio"] = grupo[f"{m}_cambio_pct"].mean()

        filas.append(fila)

    return pd.DataFrame(filas).round(3)


def tabla_14_metricas_min_max_pre_post(df, metricas=METRICAS_ARTICULO):
    """
    Tabla compacta de rangos mínimos/máximos antes y después.
    """
    filas = []

    for m in metricas:
        pre = df[f"{m}_pre"]
        post = df[f"{m}_post"]

        filas.append(
            {
                "Métrica": m,
                "Original min": pre.min(),
                "Original max": pre.max(),
                "Procesada min": post.min(),
                "Procesada max": post.max(),
                "Rango original": pre.max() - pre.min(),
                "Rango procesada": post.max() - post.min(),
            }
        )

    return pd.DataFrame(filas).round(3)


def guardar_tablas_articulo(
    df_metricas, resumen_articulo, carpeta_salida="tablas_articulo"
):
    """
    Genera y guarda todas las tablas como CSV.
    """
    from pathlib import Path

    carpeta_salida = Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    tablas = {
        "tabla_1_comparacion_pre_post": tabla_1_comparacion_pre_post(resumen_articulo),
        "tabla_2_descriptiva_original": tabla_2_descriptiva_original(df_metricas),
        "tabla_3_descriptiva_procesada": tabla_3_descriptiva_procesada(df_metricas),
        "tabla_4_robusta_mediana_iqr": tabla_4_robusta_mediana_iqr(df_metricas),
        "tabla_5_coeficiente_variacion": tabla_5_coeficiente_variacion(df_metricas),
        "tabla_6_porcentaje_imagenes_mejoradas": tabla_6_porcentaje_imagenes_mejoradas(
            df_metricas
        ),
        "tabla_7_distribucion_pipeline": tabla_7_distribucion_pipeline(df_metricas),
        "tabla_8_numero_pasos": tabla_8_numero_pasos(df_metricas),
        "tabla_9_condiciones_originales": tabla_9_condiciones_originales(df_metricas),
        "tabla_10_ranking_cambios": tabla_10_ranking_cambios(resumen_articulo),
        "tabla_11_correlacion_metricas_cambios": tabla_11_correlacion_metricas_cambios(
            df_metricas
        ),
        "tabla_12_extremos_por_metrica": tabla_12_extremos_por_metrica(
            df_metricas, n=5
        ),
        "tabla_13_resumen_por_pipeline": tabla_13_resumen_por_pipeline(df_metricas),
        "tabla_14_metricas_min_max_pre_post": tabla_14_metricas_min_max_pre_post(
            df_metricas
        ),
    }

    for nombre, tabla in tablas.items():
        ruta = carpeta_salida / f"{nombre}.csv"
        tabla.to_csv(ruta, index=False)

    print(f"Tablas guardadas en: {carpeta_salida}")

    return tablas


# ============================================================
# 7. EJECUCIÓN PRINCIPAL
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in {"features", "features_regiones"}:
        ruta_etiquetas = (
            sys.argv[2] if len(sys.argv) > 2 else "../etiquetas_regiones_SOMIB.pkl"
        )
        carpeta_features = (
            sys.argv[3]
            if len(sys.argv) > 3
            else "full_dataset"
        )

        analizar_variacion_importancia_features_regiones(
            ruta_etiquetas=ruta_etiquetas,
            carpeta_imagenes=carpeta_features,
            carpeta_salida="tablas_features",
            carpeta_figuras="figuras_metricas",
            guardar_csv=True,
            guardar_figura=True,
            mostrar=False,
        )
        raise SystemExit(0)

    # Cambia esta ruta por la carpeta real de tus imágenes
    carpeta_imagenes = "full_dataset"

    # Carpeta donde se guardarán las imágenes procesadas
    carpeta_imagenes_procesadas = "imagenes_procesadas"

    # Carpeta donde se guardarán las figuras
    carpeta_figuras = "figuras_metricas"

    # Archivos CSV de salida
    archivo_metricas = "metricas_pre_post.csv"
    archivo_resumen = "resumen_articulo.csv"

    df_metricas, resumen_articulo = comparar_original_vs_procesada(
        carpeta=carpeta_imagenes,
        salida_imagenes=carpeta_imagenes_procesadas,
        salida_csv=archivo_metricas,
        salida_resumen_csv=archivo_resumen,
        umbrales=None,
    )

    imprimir_resumen_articulo(resumen_articulo)

    estadisticas_pipeline(df_metricas)

    graficar_boxplots_pre_post(
        df_metricas,
        carpeta_salida=carpeta_figuras,
        mostrar=True,
    )

    graficar_histogramas_pre_post(
        df_metricas,
        carpeta_salida=carpeta_figuras,
        mostrar=True,
    )

    graficar_barras_cambio_pct(
        resumen_articulo,
        carpeta_salida=carpeta_figuras,
        mostrar=True,
    )
    tabla1 = tabla_comparacion_pre_post(resumen_articulo)
    tabla2 = tabla_distribucion_pipeline(df_metricas)

    print("\nTABLA 1: Comparación pre/post")
    print(tabla1.to_string(index=False))

    print("\nTABLA 2: Distribución del pipeline")
    print(tabla2.to_string(index=False))

    tabla1.to_csv("tabla_1_comparacion_pre_post.csv", index=False)
    tabla2.to_csv("tabla_2_distribucion_pipeline.csv", index=False)

    tablas = guardar_tablas_articulo(
        df_metricas=df_metricas,
        resumen_articulo=resumen_articulo,
        carpeta_salida="tablas_articulo",
    )
