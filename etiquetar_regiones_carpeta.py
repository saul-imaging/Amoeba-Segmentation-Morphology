"""
etiquetar_regiones_carpeta.py
──────────────────────────────
Recorre todas las imágenes de una carpeta, aplica la primera red (FFT + Otsu),
muestra las regiones candidatas y permite etiquetarlas manualmente como
ameba (1) o no_ameba (0).

Navegación:
  [← Anterior]  [Siguiente →]  — botones en la figura

Etiquetado:
  Tecla 1  → modo AMEBA
  Tecla 0  → modo NO_AMEBA
  Clic izquierdo sobre una región → asigna el modo activo
  u        → deshacer último clic
  c        → limpiar etiquetas de la imagen actual

Guardar:
  g  → guarda TODAS las anotaciones en el archivo .pkl y cierra

Formato del pickle guardado:
  {
    "imagen1.jpg": {
        "labels_map": array (H, W) int32,   # mapa de componentes
        "componentes": [1, 2, 5, ...],       # labels válidos (filtrados)
        "etiquetas":   {1: 1, 2: 0, 5: 1, ...},  # lab → clase
    },
    ...
  }
"""

import pickle
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from Metodo_SOMIB import segmentar_ameba_completa

# ─────────────────────────────────────────────────────────────────────────────
EXTENSIONES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades tomadas / compatibles con el pipeline original
# ─────────────────────────────────────────────────────────────────────────────

# def rellenar_huecos(mask_bin):
#     mask_u8 = mask_bin.astype(np.uint8)
#     if mask_u8.max() == 1:
#         mask_u8 = mask_u8 * 255
#     h, w = mask_u8.shape
#     flood = mask_u8.copy()
#     flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
#     cv2.floodFill(flood, flood_mask, (0, 0), 255)
#     holes = cv2.bitwise_not(flood)
#     return cv2.bitwise_or(mask_u8, holes)


# def procesar_ameba(ruta_imagen, fc_2=200):
#     """Primera red: FFT pasa-altas + Otsu + morfología."""
#     imagen = cv2.imread(str(ruta_imagen))
#     if imagen is None:
#         raise ValueError(f"No se pudo leer: {ruta_imagen}")
#     imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

#     image_fft = np.fft.fftshift(np.fft.fft2(imagen_gris))
#     rows, cols = imagen_gris.shape
#     cy, cx = rows // 2, cols // 2
#     Y, X = np.ogrid[:rows, :cols]
#     dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
#     filtro_hp = (dist >= fc_2).astype(float)

#     img_rec = np.abs(np.fft.ifft2(np.fft.ifftshift(filtro_hp * image_fft)))
#     img_u8 = cv2.normalize(img_rec, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     _, mask = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
#     return rellenar_huecos(mask)


def rellenar_huecos(mask_bin):
    mask_u8 = (mask_bin > 0).astype(np.uint8) * 255
    h, w = mask_u8.shape

    flood = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)

    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask_u8, holes)
    return (filled > 0).astype(np.uint8)


# def eliminar_componentes_en_borde(mask_bin, min_area=30):
#     """
#     Elimina componentes conectadas que toquen cualquier borde de la imagen.
#     También filtra componentes muy pequeñas.
#     """
#     mask_u8 = (mask_bin > 0).astype(np.uint8)
#     h, w = mask_u8.shape

#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
#     out = np.zeros_like(mask_u8)

#     for lab in range(1, num_labels):
#         x = stats[lab, cv2.CC_STAT_LEFT]
#         y = stats[lab, cv2.CC_STAT_TOP]
#         ww = stats[lab, cv2.CC_STAT_WIDTH]
#         hh = stats[lab, cv2.CC_STAT_HEIGHT]
#         area = stats[lab, cv2.CC_STAT_AREA]

#         toca_borde = (x == 0) or (y == 0) or (x + ww >= w) or (y + hh >= h)
#         if (not toca_borde) and (area >= min_area):
#             out[labels == lab] = 1

#     return out.astype(np.uint8)


def separar_objetos_watershed(mask_bin, imagen_gris, dist_thresh=0.22):
    mask_u8 = (mask_bin > 0).astype(np.uint8)

    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    sure_fg = (dist > dist_thresh * dist.max()).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(mask_u8, kernel, iterations=1)

    unknown = cv2.subtract(sure_bg, sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 1] = 0

    img_bgr = cv2.cvtColor(imagen_gris, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_bgr, markers)

    return markers, dist_norm


def extraer_componentes(mask_bin, min_area=80):
    mask_u8 = (mask_bin > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    componentes = [
        lab for lab in range(1, n) if stats[lab, cv2.CC_STAT_AREA] >= min_area
    ]
    return labels, stats, componentes


def obtener_centroides(labels, componentes):
    centroides = {}
    for lab in componentes:
        ys, xs = np.where(labels == lab)
        if len(ys):
            centroides[lab] = (int(ys.mean()), int(xs.mean()))
    return centroides


def _overlay_regiones(img_rgb, labels, componentes, etiquetas):
    """Construye un array RGB con las regiones coloreadas."""
    overlay = img_rgb.copy().astype(np.float32)
    for lab in componentes:
        mask = labels == lab
        cls = etiquetas.get(lab, None)
        if cls == 1:
            color = np.array([255, 50, 50], dtype=np.float32)  # rojo — ameba
        elif cls == 0:
            color = np.array([0, 220, 220], dtype=np.float32)  # cian — no_ameba
        else:
            color = np.array(
                [255, 255, 80], dtype=np.float32
            )  # amarillo — sin etiquetar
        overlay[mask] = 0.55 * overlay[mask] + 0.45 * color
    return overlay.clip(0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────


def etiquetar_regiones_carpeta(
    carpeta: str,
    archivo_salida: str = "etiquetas_regiones.pkl",
    fc_2: int = 60,
    min_area: int = 80,
    kernel_size: int = 5,
):
    """
    Recorre todas las imágenes de `carpeta`, aplica la primera red y permite
    etiquetar las regiones resultantes como ameba (1) o no_ameba (0).

    Parámetros
    ----------
    carpeta        : ruta a la carpeta con imágenes
    archivo_salida : ruta del pickle donde se guardan todos los datos
    fc_2           : radio de corte del filtro pasa-altas (primera red)
    min_area       : área mínima de componente a considerar (píxeles)
    kernel_size    : tamaño del kernel morfológico en postprocesado

    Retorna
    -------
    datos : dict  { nombre_imagen: {"labels_map", "componentes", "etiquetas"} }
    """

    # ── 1. Listar imágenes ─────────────────────────────────────────────────
    carpeta = Path(carpeta)
    archivos = sorted(p for p in carpeta.iterdir() if p.suffix.lower() in EXTENSIONES)
    if not archivos:
        raise FileNotFoundError(f"No hay imágenes en: {carpeta}")
    n_imgs = len(archivos)

    # ── 2. Estructura de datos global ──────────────────────────────────────
    # Precargamos entradas vacías; se rellenan al cargar cada imagen
    datos = {}  # nombre → {"labels_map", "componentes", "etiquetas"}
    cache_proc = {}  # nombre → (img_rgb, labels, componentes, centroides)

    # ── 3. Estado mutable ──────────────────────────────────────────────────
    estado = {
        "idx": 0,
        "modo": 1,  # 1 = ameba, 0 = no_ameba
    }

    # ── 4. Figura ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 9))
    plt.subplots_adjust(bottom=0.13)
    im_handle = ax.imshow(np.zeros((10, 10, 3), dtype=np.uint8))
    ax.axis("off")
    titulo = ax.set_title("")

    # Botones
    ax_prev = fig.add_axes([0.22, 0.02, 0.18, 0.055])
    ax_next = fig.add_axes([0.55, 0.02, 0.18, 0.055])
    btn_prev = Button(ax_prev, "← Anterior")
    btn_next = Button(ax_next, "Siguiente →")

    # ── 5. Helpers ─────────────────────────────────────────────────────────

    def _nombre():
        return archivos[estado["idx"]].name

    def _etiquetas():
        return datos[_nombre()]["etiquetas"]

    def _historial():
        return datos[_nombre()].setdefault("_hist", [])

    def _procesar_imagen(idx):
        """Carga y procesa la imagen si no está en caché."""
        nombre = archivos[idx].name
        if nombre not in cache_proc:
            ruta = archivos[idx]
            print(
                f"  Procesando [{idx + 1}/{n_imgs}]: {nombre} ...", end=" ", flush=True
            )

            # Leer imagen RGB
            img_bgr = cv2.imread(str(ruta))
            if img_bgr is None:
                print("[ERROR: no se pudo leer la imagen]")
                img_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
                labels = np.zeros((10, 10), dtype=np.int32)
                componentes = {}
                centroides = {}
                cache_proc[nombre] = (img_rgb, labels, componentes, centroides)
                if nombre not in datos:
                    datos[nombre] = {
                        "labels_map": labels.copy(),
                        "componentes": list(componentes.keys()),
                        "etiquetas": {},
                    }
                return cache_proc[nombre]

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Segmentación
            try:
                mask_raw, labels_ws, _ = segmentar_ameba_completa(img_rgb)

                # Filtrar labels por área mínima y reetiquetar
                labels = np.zeros_like(labels_ws, dtype=np.int32)
                componentes = {}
                nuevo_id = 1

                for lab in np.unique(labels_ws):
                    if lab == 0:
                        continue

                    mask_lab = labels_ws == lab
                    area = int(np.sum(mask_lab))

                    if area >= min_area:
                        labels[mask_lab] = nuevo_id
                        componentes[nuevo_id] = {"area": area}
                        nuevo_id += 1

            except Exception as e:
                print(f"[ERROR: {e}]")
                labels = np.zeros(img_rgb.shape[:2], dtype=np.int32)
                componentes = {}

            centroides = obtener_centroides(labels, componentes)
            print(f"{len(componentes)} regiones")

            cache_proc[nombre] = (img_rgb, labels, componentes, centroides)

            # Inicializar entrada de datos si no existe
            if nombre not in datos:
                datos[nombre] = {
                    "labels_map": labels.copy(),
                    "componentes": list(componentes.keys()),
                    "etiquetas": {},
                }

        return cache_proc[nombre]

    def refrescar():
        nombre = _nombre()
        img_rgb, labels, componentes, centroides = _procesar_imagen(estado["idx"])
        etiquetas = _etiquetas()

        overlay = _overlay_regiones(img_rgb, labels, componentes, etiquetas)
        im_handle.set_data(overlay)
        H, W = overlay.shape[:2]
        im_handle.set_extent([0, W, H, 0])
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

        # Números de región encima de cada centroide
        for txt in ax.texts:
            txt.remove()
        for lab, (cy, cx) in centroides.items():
            cls = etiquetas.get(lab, None)
            color_txt = "white"
            ax.text(
                cx,
                cy,
                str(lab),
                color=color_txt,
                fontsize=7,
                ha="center",
                va="center",
                bbox=dict(facecolor="black", alpha=0.45, boxstyle="round,pad=0.15"),
            )

        n_a = sum(1 for v in etiquetas.values() if v == 1)
        n_n = sum(1 for v in etiquetas.values() if v == 0)
        modo_txt = "AMEBA [1]" if estado["modo"] == 1 else "NO_AMEBA [0]"

        titulo.set_text(
            f"[{estado['idx'] + 1}/{n_imgs}]  {nombre}\n"
            f"Modo: {modo_txt}  |  Ameba: {n_a}   No_ameba: {n_n}   "
            f"Sin etiquetar: {len(componentes) - n_a - n_n}\n"
            "1=ameba | 0=no_ameba | clic=etiquetar | u=undo | c=clear | g=guardar todo"
        )

        # Leyenda compacta
        leyenda = [
            mpatches.Patch(color="#ff3232", label="Ameba (1)"),
            mpatches.Patch(color="#00dcdc", label="No ameba (0)"),
            mpatches.Patch(color="#ffff50", label="Sin etiquetar"),
        ]
        ax.legend(
            handles=leyenda,
            loc="upper right",
            fontsize=8,
            framealpha=0.6,
            handlelength=1.2,
        )

        fig.canvas.draw_idle()

    # ── 6. Callbacks ────────────────────────────────────────────────────────

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        if event.button != 1:
            return
        _, labels, componentes, _ = _procesar_imagen(estado["idx"])
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if not (0 <= x < labels.shape[1] and 0 <= y < labels.shape[0]):
            return
        lab = int(labels[y, x])
        if lab == 0 or lab not in componentes:
            return

        etiquetas = _etiquetas()
        anterior = etiquetas.get(lab, None)
        etiquetas[lab] = estado["modo"]
        _historial().append((lab, anterior))
        refrescar()

    def on_key(event):
        if event.key == "1":
            estado["modo"] = 1
        elif event.key == "0":
            estado["modo"] = 0
        elif event.key == "u":
            hist = _historial()
            if hist:
                lab, anterior = hist.pop()
                etiquetas = _etiquetas()
                if anterior is None:
                    etiquetas.pop(lab, None)
                else:
                    etiquetas[lab] = anterior
        elif event.key == "c":
            _etiquetas().clear()
            _historial().clear()
        elif event.key == "g":
            guardar_y_cerrar()
            return
        refrescar()

    def ir_anterior(event=None):
        estado["idx"] = (estado["idx"] - 1) % n_imgs
        refrescar()

    def ir_siguiente(event=None):
        estado["idx"] = (estado["idx"] + 1) % n_imgs
        refrescar()

    def guardar_y_cerrar(event=None):
        # Limpiar claves internas de historial antes de guardar
        datos_limpios = {}
        for nombre, d in datos.items():
            datos_limpios[nombre] = {
                "labels_map": d["labels_map"],
                "componentes": d["componentes"],
                "etiquetas": d["etiquetas"],
            }

        ruta_salida = Path(archivo_salida)
        with open(ruta_salida, "wb") as f:
            pickle.dump(datos_limpios, f)

        n_anotadas = sum(1 for d in datos_limpios.values() if d["etiquetas"])
        print(f"\n[✓] Guardado en: {ruta_salida.resolve()}")
        print(f"    Imágenes con al menos 1 región etiquetada: {n_anotadas} / {n_imgs}")
        plt.close(fig)

    # ── 7. Conectar eventos ─────────────────────────────────────────────────
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    btn_prev.on_clicked(ir_anterior)
    btn_next.on_clicked(ir_siguiente)

    # ── 8. Arrancar ─────────────────────────────────────────────────────────
    refrescar()
    plt.show(block=True)

    # Limpiar historiales y devolver
    return {
        nombre: {k: v for k, v in d.items() if k != "_hist"}
        for nombre, d in datos.items()
    }


def preparar_dataset_segunda_red(
    datos: dict,
    carpeta: str,
    val_frac: float = 0.25,
    seed: int = 0,
) -> dict:
    """
    A partir del dict `datos` devuelto por `etiquetar_regiones_carpeta`,
    ejecuta los pasos 3-5 del pipeline original y devuelve todo listo
    para entrenar la segunda red.

    Pasos ejecutados
    ----------------
    3) construir_dataset_segunda_red_desde_regiones  (todas las imágenes)
    4) normalizar_features_objeto
    5) split_dataset_objetos

    Parámetros
    ----------
    datos    : dict devuelto por etiquetar_regiones_carpeta (o cargar_etiquetas)
    carpeta  : carpeta original de imágenes (necesaria para re-leer los RGB)
    val_frac : fracción de validación (default 0.25)
    seed     : semilla para el split (default 0)

    Retorna
    -------
    dict con las claves:
        X_obj, y_obj           — features crudas y etiquetas (N,)
        X_obj_n                — features normalizadas
        mu_obj, sigma_obj      — parámetros de normalización
        Xtr, ytr               — train
        Xval, yval             — val  (= train si val quedó vacío)
        labs_validos           — lista de (nombre_imagen, lab) por fila
    """
    carpeta = Path(carpeta)

    # ── Paso 3: extraer features de todas las regiones etiquetadas ──────────
    print("\n[Paso 3] Extrayendo features de objetos...")
    X_list, y_list, labs_validos = [], [], []

    for nombre, d in datos.items():
        etiquetas = d["etiquetas"]
        if not etiquetas:
            continue

        labels_map = d["labels_map"]
        ruta = carpeta / nombre
        img_bgr = cv2.imread(str(ruta))
        if img_bgr is None:
            print(f"  [!] No se pudo leer {nombre} — saltando")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        for lab, cls in etiquetas.items():
            feat = _extraer_features_objeto(img_rgb, labels_map, int(lab))
            X_list.append(feat)
            y_list.append(int(cls))
            labs_validos.append((nombre, int(lab)))

    if not X_list:
        raise ValueError(
            "No hay regiones etiquetadas en ninguna imagen. "
            "Etiqueta al menos una ameba (1) y un no_ameba (0) antes de continuar."
        )

    X_obj = np.vstack(X_list).astype(np.float32)
    y_obj = np.array(y_list, dtype=np.int32)

    print(f"  X_obj.shape : {X_obj.shape}")
    clases, conteos = np.unique(y_obj, return_counts=True)
    for cls, cnt in zip(clases, conteos):
        nombre_cls = "ameba" if cls == 1 else "no_ameba"
        print(f"  Clase {cls} ({nombre_cls}) : {cnt} muestras")

    if len(clases) < 2:
        raise ValueError(
            "La segunda red necesita al menos 2 clases: ameba (1) y no_ameba (0). "
            f"Solo se encontró la clase: {clases.tolist()}"
        )

    # ── Paso 4: normalización ───────────────────────────────────────────────
    print("\n[Paso 4] Normalizando features...")
    mu_obj = X_obj.mean(axis=0)
    sigma_obj = X_obj.std(axis=0)
    sigma_obj = np.where(sigma_obj < 1e-8, 1.0, sigma_obj)
    X_obj_n = ((X_obj - mu_obj) / sigma_obj).astype(np.float32)

    print(f"  mu    : {np.round(mu_obj, 4)}")
    print(f"  sigma : {np.round(sigma_obj, 4)}")

    # ── Paso 5: split train / val ────────────────────────────────────────────
    print(f"\n[Paso 5] Split train/val  (val_frac={val_frac}, seed={seed})...")
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X_obj_n))
    rng.shuffle(idx)
    n_val = max(1, min(len(idx) - 1, int(round(len(idx) * val_frac))))

    idx_val = idx[:n_val]
    idx_tr = idx[n_val:]

    Xtr, ytr = X_obj_n[idx_tr], y_obj[idx_tr]
    Xval, yval = X_obj_n[idx_val], y_obj[idx_val]

    # Salvaguarda: si val quedó sin todas las clases, usar train como respaldo
    if len(np.unique(yval)) < 2:
        print("  [!] Val no tiene todas las clases — usando train como val")
        Xval, yval = Xtr.copy(), ytr.copy()

    print(f"  Train : {len(Xtr)} muestras")
    print(f"  Val   : {len(Xval)} muestras")

    return {
        "X_obj": X_obj,
        "y_obj": y_obj,
        "X_obj_n": X_obj_n,
        "mu_obj": mu_obj.astype(np.float32),
        "sigma_obj": sigma_obj.astype(np.float32),
        "Xtr": Xtr,
        "ytr": ytr,
        "Xval": Xval,
        "yval": yval,
        "labs_validos": labs_validos,
    }


FEATURES_OBJETO = [
    # Forma/topología mínima
    "solidez",
    "num_huecos",
    "euler_number",
    # Intensidad interna robusta
    "intensidad_std",
    "intensidad_cv",
    "intensidad_iqr",
    "entropia_intensidad",
    # Contraste contra entorno cercano
    "contraste_media_objeto_fondo",
    "contraste_std_objeto_fondo",
    "contraste_entropia_objeto_fondo",
    # Textura GLCM
    "glcm_contrast",
    "glcm_dissimilarity",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
    # Textura local LBP
    "lbp_uniformidad",
    "lbp_entropia",
    # Frecuencia
    "fft_ratio",
    "spectral_entropy",
    "energia_baja",
    "energia_media",
    "energia_alta",
    # Bordes / estructura interna
    "edge_density",
    "gradiente_medio",
    "gradiente_std",
    "laplacian_var",
    "log_response_mean",
    "log_response_std",
    # Contexto espacial
    "toca_borde",
    "distancia_borde_norm",
]


def _entropia_vector_uint8(vals):
    EPS = 1e-8

    if vals is None or len(vals) == 0:
        return 0.0

    vals = np.asarray(vals, dtype=np.uint8)
    hist = np.bincount(vals, minlength=256).astype(np.float32)
    p = hist / (hist.sum() + EPS)
    p = p[p > 0]

    return float(-np.sum(p * np.log2(p + EPS)))


def _contar_huecos_y_euler(comp_mask):
    """
    Calcula número de huecos internos y número de Euler aproximado.
    Euler = componentes - huecos.
    """
    comp_mask = (comp_mask > 0).astype(np.uint8)

    n_comp, _ = cv2.connectedComponents(comp_mask, connectivity=8)
    num_componentes = max(0, n_comp - 1)

    h, w = comp_mask.shape
    fondo = (comp_mask == 0).astype(np.uint8) * 255

    flood = fondo.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    cv2.floodFill(flood, flood_mask, (0, 0), 128)

    # Huecos = fondo que no fue alcanzado desde el exterior
    holes_mask = ((flood == 255) & (comp_mask == 0)).astype(np.uint8)

    n_holes, _ = cv2.connectedComponents(holes_mask, connectivity=8)
    num_huecos = max(0, n_holes - 1)

    euler_number = float(num_componentes - num_huecos)

    return float(num_huecos), euler_number


def _extraer_features_objeto_core(img_rgb, labels, lab, context_pad=12):
    """
    Extractor común para entrenamiento e inferencia.

    No usa como features:
    - area
    - perimetro
    - circularidad
    - aspect_ratio
    - extent
    - momentos de Hu
    """
    EPS = 1e-8
    N_FEATURES = len(FEATURES_OBJETO)

    if img_rgb is None or labels is None:
        return np.zeros(N_FEATURES, dtype=np.float32)

    if img_rgb.ndim == 2:
        img_gray = img_rgb.astype(np.uint8)
    else:
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    H, W = labels.shape[:2]

    ys, xs = np.where(labels == lab)
    if len(ys) == 0 or len(xs) == 0:
        return np.zeros(N_FEATURES, dtype=np.float32)

    # Bounding box original del objeto
    y_min, y_max = int(ys.min()), int(ys.max()) + 1
    x_min, x_max = int(xs.min()), int(xs.max()) + 1

    # Bounding box con contexto alrededor
    y0 = max(0, y_min - context_pad)
    y1 = min(H, y_max + context_pad)
    x0 = max(0, x_min - context_pad)
    x1 = min(W, x_max + context_pad)

    crop_gray = img_gray[y0:y1, x0:x1].astype(np.uint8)
    crop_labels = labels[y0:y1, x0:x1]
    comp_mask = (crop_labels == lab).astype(np.uint8)

    vals_obj = crop_gray[comp_mask > 0]

    if vals_obj.size == 0:
        return np.zeros(N_FEATURES, dtype=np.float32)

    # =========================================================
    # 1) FORMA / TOPOLOGÍA MÍNIMA
    # =========================================================
    contornos, _ = cv2.findContours(
        comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contornos:
        cnt = max(contornos, key=cv2.contourArea)
        area_contorno = float(cv2.contourArea(cnt))

        hull = cv2.convexHull(cnt)
        area_hull = float(cv2.contourArea(hull))

        solidez = float(area_contorno / (area_hull + EPS)) if area_hull > EPS else 0.0
    else:
        solidez = 0.0

    num_huecos, euler_number = _contar_huecos_y_euler(comp_mask)

    # =========================================================
    # 2) INTENSIDAD INTERNA ROBUSTA
    # =========================================================
    intensidad_mean = float(np.mean(vals_obj))
    intensidad_std = float(np.std(vals_obj))
    intensidad_cv = float(intensidad_std / (abs(intensidad_mean) + EPS))

    q25, q75 = np.percentile(vals_obj, [25, 75])
    intensidad_iqr = float(q75 - q25)

    entropia_intensidad = _entropia_vector_uint8(vals_obj)

    # =========================================================
    # 3) CONTRASTE OBJETO VS FONDO CERCANO
    # =========================================================
    kernel_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilatada = cv2.dilate(comp_mask, kernel_ring, iterations=1)

    ring_mask = ((dilatada > 0) & (comp_mask == 0)).astype(np.uint8)
    vals_ring = crop_gray[ring_mask > 0]

    if vals_ring.size > 0:
        media_ring = float(np.mean(vals_ring))
        std_ring = float(np.std(vals_ring))
        ent_ring = _entropia_vector_uint8(vals_ring)

        contraste_media_objeto_fondo = float(intensidad_mean - media_ring)
        contraste_std_objeto_fondo = float(intensidad_std / (std_ring + EPS))
        contraste_entropia_objeto_fondo = float(entropia_intensidad - ent_ring)
    else:
        contraste_media_objeto_fondo = 0.0
        contraste_std_objeto_fondo = 0.0
        contraste_entropia_objeto_fondo = 0.0

    # =========================================================
    # 4) TEXTURA GLCM
    # =========================================================
    crop_filled = crop_gray.copy()
    mediana_obj = int(np.median(vals_obj))
    crop_filled[comp_mask == 0] = mediana_obj

    levels = 32
    crop_q = (crop_filled.astype(np.float32) / 256.0 * levels).astype(np.uint8)
    crop_q = np.clip(crop_q, 0, levels - 1)

    try:
        glcm = graycomatrix(
            crop_q,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=levels,
            symmetric=True,
            normed=True,
        )

        glcm_contrast = float(np.mean(graycoprops(glcm, "contrast")))
        glcm_dissimilarity = float(np.mean(graycoprops(glcm, "dissimilarity")))
        glcm_homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
        glcm_energy = float(np.mean(graycoprops(glcm, "energy")))
        glcm_correlation = float(np.mean(graycoprops(glcm, "correlation")))

        if not np.isfinite(glcm_correlation):
            glcm_correlation = 0.0

    except Exception:
        glcm_contrast = 0.0
        glcm_dissimilarity = 0.0
        glcm_homogeneity = 0.0
        glcm_energy = 0.0
        glcm_correlation = 0.0

    # =========================================================
    # 5) TEXTURA LOCAL LBP
    # =========================================================
    try:
        P = 8
        R = 1

        lbp = local_binary_pattern(crop_filled, P=P, R=R, method="uniform")
        lbp_vals = lbp[comp_mask > 0].astype(np.int32)

        n_bins = P + 2
        hist_lbp = np.bincount(lbp_vals, minlength=n_bins).astype(np.float32)
        p_lbp = hist_lbp / (hist_lbp.sum() + EPS)

        lbp_uniformidad = float(np.sum(p_lbp**2))

        p_nonzero = p_lbp[p_lbp > 0]
        lbp_entropia = float(-np.sum(p_nonzero * np.log2(p_nonzero + EPS)))

    except Exception:
        lbp_uniformidad = 0.0
        lbp_entropia = 0.0

    # =========================================================
    # 6) FRECUENCIA: FFT RATIO + ENTROPÍA + BANDAS
    # =========================================================
    patch = crop_gray.astype(np.float32) * comp_mask.astype(np.float32)
    patch = (patch - intensidad_mean) * comp_mask.astype(np.float32)

    if patch.size > 0 and np.sum(comp_mask) > 0:
        F = np.fft.fftshift(np.fft.fft2(patch))
        E = np.abs(F) ** 2

        h, w = E.shape
        yy, xx = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

        rmax = float(rr.max() + EPS)

        r1 = 0.18 * rmax
        r2 = 0.45 * rmax

        low_mask = rr <= r1
        mid_mask = (rr > r1) & (rr <= r2)
        high_mask = rr > r2

        e_low = float(E[low_mask].sum())
        e_mid = float(E[mid_mask].sum())
        e_high = float(E[high_mask].sum())
        e_total = e_low + e_mid + e_high + EPS

        fft_ratio = float((e_mid + e_high) / (e_low + EPS))

        energia_baja = float(e_low / e_total)
        energia_media = float(e_mid / e_total)
        energia_alta = float(e_high / e_total)

        E_flat = E.ravel().astype(np.float64)
        E_sum = float(E_flat.sum())

        if E_sum > EPS:
            ps = E_flat / E_sum
            ps = ps[ps > 0]
            spectral_entropy = float(-np.sum(ps * np.log2(ps + EPS)))
            spectral_entropy /= float(np.log2(len(E_flat) + EPS))
        else:
            spectral_entropy = 0.0

    else:
        fft_ratio = 0.0
        spectral_entropy = 0.0
        energia_baja = 0.0
        energia_media = 0.0
        energia_alta = 0.0

    # =========================================================
    # 7) BORDES / GRADIENTE / LAPLACIANO / LoG
    # =========================================================
    edges = cv2.Canny(crop_filled, 50, 150)
    edge_density = float(
        np.sum((edges > 0) & (comp_mask > 0)) / (np.sum(comp_mask) + EPS)
    )

    gx = cv2.Sobel(crop_filled, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop_filled, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)

    grad_vals = grad_mag[comp_mask > 0]
    if grad_vals.size > 0:
        gradiente_medio = float(np.mean(grad_vals))
        gradiente_std = float(np.std(grad_vals))
    else:
        gradiente_medio = 0.0
        gradiente_std = 0.0

    lap = cv2.Laplacian(crop_filled, cv2.CV_32F)
    lap_vals = lap[comp_mask > 0]
    laplacian_var = float(np.var(lap_vals)) if lap_vals.size > 0 else 0.0

    blur_log = cv2.GaussianBlur(crop_filled, (0, 0), sigmaX=1.2)
    log_response = cv2.Laplacian(blur_log, cv2.CV_32F)
    log_vals = np.abs(log_response[comp_mask > 0])

    if log_vals.size > 0:
        log_response_mean = float(np.mean(log_vals))
        log_response_std = float(np.std(log_vals))
    else:
        log_response_mean = 0.0
        log_response_std = 0.0

    # =========================================================
    # 8) CONTEXTO ESPACIAL
    # =========================================================
    margen_borde = 3

    toca_borde_bool = (
        x_min <= margen_borde
        or y_min <= margen_borde
        or x_max >= W - margen_borde
        or y_max >= H - margen_borde
    )

    toca_borde = 1.0 if toca_borde_bool else 0.0

    distancia_borde = min(
        x_min,
        y_min,
        W - x_max,
        H - y_max,
    )

    distancia_borde_norm = float(distancia_borde / (min(H, W) + EPS))

    # =========================================================
    # VECTOR FINAL
    # =========================================================
    features = np.array(
        [
            solidez,
            num_huecos,
            euler_number,
            intensidad_std,
            intensidad_cv,
            intensidad_iqr,
            entropia_intensidad,
            contraste_media_objeto_fondo,
            contraste_std_objeto_fondo,
            contraste_entropia_objeto_fondo,
            glcm_contrast,
            glcm_dissimilarity,
            glcm_homogeneity,
            glcm_energy,
            glcm_correlation,
            lbp_uniformidad,
            lbp_entropia,
            fft_ratio,
            spectral_entropy,
            energia_baja,
            energia_media,
            energia_alta,
            edge_density,
            gradiente_medio,
            gradiente_std,
            laplacian_var,
            log_response_mean,
            log_response_std,
            toca_borde,
            distancia_borde_norm,
        ],
        dtype=np.float32,
    )

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def _extraer_features_objeto(img_rgb, labels, lab):
    return _extraer_features_objeto_core(img_rgb, labels, lab)


def cargar_etiquetas(ruta_pkl: str) -> dict:
    """
    Lee el archivo guardado y devuelve el dict completo.

    Uso:
        datos = cargar_etiquetas("etiquetas_regiones.pkl")
        etiquetas_img1 = datos["ameba2507_0110.jpg"]["etiquetas"]
        # {3: 1, 7: 0, 12: 1, ...}
    """
    with open(ruta_pkl, "rb") as f:
        return pickle.load(f)


def resumen_etiquetas(ruta_pkl: str):
    """Imprime un resumen legible del archivo guardado."""
    datos = cargar_etiquetas(ruta_pkl)
    print(f"{'Imagen':<35} {'Regiones':>8} {'Ameba':>7} {'No_ameba':>10}")
    print("─" * 62)
    for nombre, d in datos.items():
        et = d["etiquetas"]
        n_a = sum(1 for v in et.values() if v == 1)
        n_n = sum(1 for v in et.values() if v == 0)
        print(f"{nombre:<35} {len(d['componentes']):>8} {n_a:>7} {n_n:>10}")


# ─────────────────────────────────────────────────────────────────────────────
# Ejemplo de uso
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    carpeta = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "GAYM_25_07_25-20251219T121604Z-1-001 1/GAYM_25_07_25/images1"
    )
    salida = sys.argv[2] if len(sys.argv) > 2 else "etiquetas_regiones_SOMIB.pkl"

    datos = etiquetar_regiones_carpeta(
        carpeta=carpeta, archivo_salida=salida, fc_2=50, min_area=10, kernel_size=5
    )

    print("\nResumen final:")
    resumen_etiquetas(salida)
