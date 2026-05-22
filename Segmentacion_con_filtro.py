import math
import os
import pickle
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from etiquetar_regiones_carpeta import (
    cargar_etiquetas,
    etiquetar_regiones_carpeta,
    preparar_dataset_segunda_red,
)
from Metodo_SOMIB import segmentar_ameba_completa


def init_prototypes_from_samples(X, y, total=200, seed=0, noise_frac=0.05):
    """
    Inicializa prototipos para todas las clases presentes en y.
    Reparte los prototipos de forma casi uniforme entre clases.
    """
    rng = np.random.default_rng(seed)

    clases = np.unique(y)
    if len(clases) < 2:
        raise ValueError("Se necesitan al menos 2 clases para inicializar prototipos.")

    base = total // len(clases)
    resto = total % len(clases)

    W_list = []
    w2_list = []

    for i, cls in enumerate(clases):
        idx = np.where(y == cls)[0]
        if len(idx) == 0:
            continue

        n_cls = base + (1 if i < resto else 0)
        sel = rng.choice(idx, size=n_cls, replace=(len(idx) < n_cls))
        Wc = X[sel].copy()

        # ruido suave para evitar prototipos idénticos
        sigma = X[idx].std(axis=0)
        sigma = np.where(sigma < 1e-6, 0.01, sigma)
        Wc = Wc + rng.normal(0.0, noise_frac * sigma, size=Wc.shape)

        # recortar al rango de la clase
        lo = X[idx].min(axis=0)
        hi = X[idx].max(axis=0)
        Wc = np.clip(Wc, lo, hi)

        W_list.append(Wc.astype(np.float32))
        w2_list.extend([cls] * len(Wc))

    if len(W_list) == 0:
        raise ValueError("No se pudieron inicializar prototipos.")

    W = np.vstack(W_list).astype(np.float32)
    w2 = np.array(w2_list, dtype=np.int32)

    return W, w2


# 7) LÉVY STEP + ALPHA


def levy_step(beta=1.5, rng=random):
    sigma_u = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = rng.gauss(0, sigma_u)
    v = rng.gauss(0, 1.0)
    return u / (abs(v) ** (1.0 / beta))


def levy_alpha(
    t,
    T,
    alpha0=0.3,
    alpha_min=0.01,
    p=2.0,
    p_jump=0.12,
    k=0.08,
    beta=1.5,
    freeze_frac=0.7,
    rng=None,
):
    if rng is None:
        rng = random
    frac = max(0.0, 1.0 - t / max(1, T))
    alpha = alpha0 * (frac**p)
    if (t / max(1, T)) < freeze_frac and rng.random() < p_jump:
        jump = abs(levy_step(beta, rng))
        alpha = alpha * (1.0 + k * jump)
    return max(alpha_min, min(alpha, alpha0))


# 8) PREDICCIÓN / ACCURACY / RESEED
def predict_labels(W, X, w2):
    X2 = np.sum(X * X, axis=1, keepdims=True)
    W2 = np.sum(W * W, axis=1)[None, :]
    dist2 = X2 + W2 - 2.0 * (X @ W.T)
    return w2[np.argmin(dist2, axis=1)]


def accuracy(W, X, y, w2):
    if len(X) == 0:
        return 0.0
    return float(np.mean(predict_labels(W, X, w2) == y))


def make_reseed_fn(W_ref, w2_ref, X_ref, y_ref, seed_np=0, reseed_noise=0.08):
    rng = np.random.default_rng(seed_np)

    def reseed_neurona(j):
        cls = int(w2_ref[j])
        idx_data = np.flatnonzero(y_ref == cls)

        if idx_data.size >= 2:
            Xc = X_ref[idx_data]
            lo, hi = Xc.min(axis=0), Xc.max(axis=0)
            if np.isfinite(lo).all() and np.isfinite(hi).all() and np.all(hi > lo):
                W_ref[j, :] = rng.uniform(lo, hi).astype(np.float32)
                return "bbox-data"

        if idx_data.size > 0:
            x = X_ref[rng.choice(idx_data)]
            W_ref[j, :] = (
                x + rng.normal(0.0, reseed_noise, size=W_ref.shape[1])
            ).astype(np.float32)
            return "class-rand"

        return "noop"

    return reseed_neurona


# 9) ENTRENAMIENTO HLVQ PARA SEGMENTACIÓN
def train_hybrid_lvq_segmentacion(
    Xtr,
    ytr,
    Xval,
    yval,
    *,
    W_init,
    w2_init,
    epocas=300,
    alpha0=0.03,
    alpha_min=0.001,
    levy_p=2.0,
    levy_p_jump=0.12,
    levy_k=0.08,
    levy_beta=1.5,
    levy_freeze_frac=0.7,
    alphaerror=0.25,
    n_media_max=7,
    dropout_p=0.15,
    dropout_warmup=10,
    dropout_stop_frac=0.5,
    ensure_one_per_class=True,
    reseed_noise=0.08,
    eval_every=1,
    verbose_every=25,
    patience=None,
    min_delta=0.0,
    seed_np=0,
    seed_py=7,
    return_last=False,
    guardar_cada=1,
):
    rng_np = np.random.default_rng(seed_np)
    py_rng = random.Random(seed_py)

    W = W_init.copy().astype(np.float32)
    w2 = w2_init.copy().astype(np.int32)

    reseed_neurona = make_reseed_fn(
        W, w2, Xtr, ytr, seed_np=seed_np, reseed_noise=reseed_noise
    )
    class_to_idx = {c: np.flatnonzero(w2 == c) for c in np.unique(w2)}

    best_acc = -1.0
    best_W = W.copy()
    best_epoch = 0

    hist_train = []
    hist_val = []
    history_W = []
    history_epochs = []

    bad = 0
    total = W.shape[0]

    for t in range(epocas):
        alpha = levy_alpha(
            t,
            epocas,
            alpha0=alpha0,
            alpha_min=alpha_min,
            p=levy_p,
            p_jump=levy_p_jump,
            k=levy_k,
            beta=levy_beta,
            freeze_frac=levy_freeze_frac,
            rng=py_rng,
        )

        asignaciones = [[] for _ in range(total)]
        idx_perm = rng_np.permutation(Xtr.shape[0])

        if (
            t < dropout_warmup
            or dropout_p <= 0.0
            or (t / max(1, epocas)) >= dropout_stop_frac
        ):
            active = np.ones(total, dtype=bool)
        else:
            active = rng_np.random(total) < (1.0 - dropout_p)
            if ensure_one_per_class:
                for c in np.unique(w2):
                    idx_c = class_to_idx.get(c, np.array([], dtype=int))
                    if idx_c.size > 0 and not np.any(active[idx_c]):
                        active[rng_np.choice(idx_c)] = True

        for ii in idx_perm:
            x = Xtr[ii]
            yi = ytr[ii]

            dist2 = np.sum((W - x) ** 2, axis=1)
            dist2_masked = np.where(active, dist2, np.inf)
            g = int(np.argmin(dist2_masked))

            if w2[g] == yi:
                asignaciones[g].append(ii)
            else:
                W[g] -= (alphaerror * alpha) * (x - W[g])

        for j in range(total):
            idxs = asignaciones[j]
            if len(idxs) == 0:
                reseed_neurona(j)
                continue

            Xj = Xtr[idxs]
            dists = np.sum((Xj - W[j]) ** 2, axis=1)
            top_k = np.argsort(dists)[: min(len(dists), n_media_max)]
            W[j] = W[j] + alpha * (Xj[top_k].mean(axis=0) - W[j])

        if (t + 1) % guardar_cada == 0 or t == epocas - 1:
            history_W.append(W.copy())
            history_epochs.append(t + 1)

        if (t + 1) % eval_every == 0 or t == epocas - 1:
            acc_tr = accuracy(W, Xtr, ytr, w2)
            acc_val = accuracy(W, Xval, yval, w2)

            hist_train.append(acc_tr)
            hist_val.append(acc_val)

            if acc_val > (best_acc + min_delta):
                best_acc = acc_val
                best_W = W.copy()
                best_epoch = t + 1
                bad = 0
            else:
                bad += 1

            if verbose_every and ((t + 1) % verbose_every == 0 or t == epocas - 1):
                print(
                    f"época {t + 1:4d} | train={acc_tr:.4f} | val={acc_val:.4f} | "
                    f"best_val={best_acc:.4f} (época {best_epoch}) | bad={bad}/{patience}"
                )

            if patience and bad >= patience:
                break

    out = {
        "name": "HybridLVQ-Segmentacion",
        "W_best": best_W,
        "w2": w2,
        "best_acc": float(best_acc),
        "best_epoch": int(best_epoch),
        "hist_train": np.array(hist_train, dtype=float),
        "hist_val": np.array(hist_val, dtype=float),
        "stopped_epoch": int(t + 1),
        "history_W": history_W,
        "history_epochs": history_epochs,
    }

    if return_last:
        out["W_last"] = W.copy()
        out["last_acc"] = float(accuracy(W, Xval, yval, w2))
        out["last_epoch"] = int(t + 1)

    return out


def mostrar_segmentacion_sin_semillas(img_rgb, mask_raw, mascaras_finales):
    """
    mascaras_finales: dict  {nombre: (mask_uint8, color_rgb)}
    Ej: {"ameba":   (mask_ameba,   (255,  50,  50)),
         "cristal": (mask_cristal, (255, 220,   0)),
         "residuo": (mask_residuo, (255, 165,   0))}
    """
    # Overlay compuesto con todas las clases
    overlay = img_rgb.copy()
    for nombre, (mask, color) in mascaras_finales.items():
        c = np.array(color, dtype=np.float32)
        overlay[mask > 0] = (
            (0.55 * overlay[mask > 0] + 0.45 * c).clip(0, 255).astype(np.uint8)
        )

    # Mapa de clases en color para mask_raw
    paleta = {
        0: (30, 30, 30),  # fondo  → gris oscuro
        1: (220, 50, 50),  # ameba  → rojo
        2: (220, 210, 40),  # cristal → amarillo
        3: (255, 140, 0),  # residuo → naranja
    }
    mapa_color = np.zeros((*mask_raw.shape, 3), dtype=np.uint8)
    for cls, color in paleta.items():
        mapa_color[mask_raw == cls] = color

    ncols = 2 + len(mascaras_finales)  # original + crudo + una por clase
    _, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Imagen original")
    axes[0].axis("off")

    axes[1].imshow(mapa_color)
    axes[1].set_title("Segmentación cruda")
    axes[1].axis("off")

    for i, (nombre, (mask, color)) in enumerate(mascaras_finales.items(), start=2):
        single = img_rgb.copy()
        c = np.array(color, dtype=np.float32)
        single[mask > 0] = (
            (0.55 * single[mask > 0] + 0.45 * c).clip(0, 255).astype(np.uint8)
        )
        axes[i].imshow(single)
        axes[i].set_title(f"{nombre.capitalize()} final")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def rellenar_huecos(mask_bin):
    """
    Rellena los huecos internos de una máscara binaria usando floodFill.
    """
    # Asegurarnos de que la máscara sea de 8 bits (0 o 255)
    mask_u8 = mask_bin.astype(np.uint8)
    if mask_u8.max() == 1:
        mask_u8 = mask_u8 * 255

    h, w = mask_u8.shape
    flood = mask_u8.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # Asumimos que la esquina (0,0) es fondo. Rellenamos el fondo exterior.
    cv2.floodFill(flood, flood_mask, (0, 0), 255)

    # Invertimos el fondo rellenado para obtener solo los huecos internos
    holes = cv2.bitwise_not(flood)

    # Combinamos la máscara original con los huecos rellenados
    filled = cv2.bitwise_or(mask_u8, holes)
    return filled


def separar_objetos_watershed(mask_bin, gray_img, dist_thresh=0.35):
    """
    mask_bin: máscara binaria 0/255 o 0/1
    gray_img: imagen original en gris
    """
    mask_u8 = (mask_bin > 0).astype(np.uint8)

    # Distancia al fondo
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)

    # Normalizar solo para visualización si quieres
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # Picos seguros del foreground
    sure_fg = (dist > dist_thresh * dist.max()).astype(np.uint8)

    # Fondo seguro
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(mask_u8, kernel, iterations=1)

    # Región desconocida
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marcadores
    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 1] = 0

    # Watershed necesita imagen BGR
    img_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_bgr, markers)

    # Cada objeto queda con label > 1
    return markers, dist_norm


def procesar_ameba(ruta_imagen, fc_2=80):
    """
    Procesa la imagen de una ameba aplicando un filtro pasa altas en el dominio
    de la frecuencia y operaciones morfológicas.
    """
    # 1. Leer la imagen y validar que exista
    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No se encontró la imagen en: {ruta_imagen}")

    imagen = cv2.imread(ruta_imagen)
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # 2. Transformada de Fourier (FFT)
    image_fft = np.fft.fftshift(np.fft.fft2(imagen_gris))

    rows, cols = imagen_gris.shape
    center_row, center_col = rows // 2, cols // 2

    # 3. Creación del Filtro Pasa Altas (Vectorizado)
    Y, X = np.ogrid[:rows, :cols]
    distancia_al_centro = np.sqrt((X - center_col) ** 2 + (Y - center_row) ** 2)
    filtro_hp = (distancia_al_centro >= fc_2).astype(float)

    # 4. Aplicar el filtro y calcular la FFT inversa
    convolucion_hp = filtro_hp * image_fft
    real_hp = np.fft.ifft2(np.fft.ifftshift(convolucion_hp))
    img_new_hp = np.abs(real_hp)

    img_normalizada = cv2.normalize(img_new_hp, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    _, mask_o = cv2.threshold(
        img_normalizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel_size = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    mask = cv2.morphologyEx(mask_o, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # 7. Rellenar huecos
    mask_final = rellenar_huecos(mask)

    markers, dist_norm = separar_objetos_watershed(
        mask_final, imagen_gris, dist_thresh=0.5
    )

    return imagen_gris, mask_final


def superponer_mascara(imagen_gris, mascara_bin, color=(0, 255, 0), alpha=0.4):
    """
    Superpone una máscara binaria sobre la imagen original con transparencia.

    Args:
        imagen_gris: Imagen original en escala de grises.
        mascara_bin: Máscara binaria (0 o 255).
        color: Tupla BGR para el color de la superposición (Verde por defecto).
        alpha: Transparencia del color (0.0 transparente, 1.0 opaco).

    Returns:
        Imagen a color con la máscara superpuesta.
    """
    # 1. Convertir la imagen gris a BGR (color) para poder pintarla
    img_color = cv2.cvtColor(imagen_gris, cv2.COLOR_GRAY2BGR)

    # 2. Crear una imagen del color sólido elegido
    res_color = np.zeros_like(img_color)
    res_color[:] = color

    # 3. Crear una máscara de 3 canales a partir de la binaria para usarla como bitmask
    bitmask = cv2.cvtColor(mascara_bin, cv2.COLOR_GRAY2BGR)

    # 4. Aislar el color solo en la zona de la ameba detectada
    res_color = cv2.bitwise_and(res_color, bitmask)

    # 5. Mezclar la imagen original con la zona coloreada (Blending)
    # imagen_final = original * 1 + zona_color * alpha + 0
    imagen_final = cv2.addWeighted(img_color, 1.0, res_color, alpha, 0)

    # Convertir a RGB para mostrar correctamente en Matplotlib
    return cv2.cvtColor(imagen_final, cv2.COLOR_BGR2RGB)


# =============================================================================
# SEGUNDA RED: CLASIFICACIÓN POR OBJETO
# =============================================================================


def extraer_componentes(mask_bin, min_area=80):
    """
    mask_bin: máscara binaria uint8 (0/1)
    Devuelve:
      labels       : mapa de etiquetas de componentes
      stats        : estadísticas de connectedComponentsWithStats
      componentes  : lista de labels válidos (filtrados por área)
    """
    mask_u8 = (mask_bin > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    componentes = []
    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area >= min_area:
            componentes.append(lab)

    return labels, stats, componentes


def recortar_componente(labels, lab, img_gray):
    ys, xs = np.where(labels == lab)

    if len(ys) == 0 or len(xs) == 0:
        return None, None, None

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    comp_mask = (labels[y0:y1, x0:x1] == lab).astype(np.uint8)
    crop_gray = img_gray[y0:y1, x0:x1]

    return comp_mask, crop_gray, (y0, y1, x0, x1)


def calcular_circularidad(comp_mask):
    contornos, _ = cv2.findContours(
        comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contornos) == 0:
        return 0.0

    cnt = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimetro = cv2.arcLength(cnt, True)

    if perimetro <= 1e-8:
        return 0.0

    circularidad = (4.0 * np.pi * area) / (perimetro**2 + 1e-8)
    return float(circularidad)


def calcular_solidez(comp_mask):
    contornos, _ = cv2.findContours(
        comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contornos) == 0:
        return 0.0

    cnt = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area <= 1e-8:
        return 0.0

    hull = cv2.convexHull(cnt)
    area_hull = cv2.contourArea(hull)
    if area_hull <= 1e-8:
        return 0.0

    return float(area / area_hull)


def calcular_fft_ratio(crop_gray, comp_mask, frac_radio_bajo=0.18):
    """
    Ratio de energía alta/baja frecuencia.
    """
    patch = crop_gray.astype(np.float32) * comp_mask.astype(np.float32)

    if patch.size == 0 or np.sum(comp_mask) == 0:
        return 0.0

    valores = patch[comp_mask > 0]
    media = valores.mean() if valores.size > 0 else 0.0
    patch = patch - media
    patch *= comp_mask

    F = np.fft.fft2(patch)
    F = np.fft.fftshift(F)
    energia = np.abs(F) ** 2

    h, w = energia.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    rmax = rr.max() + 1e-8
    rbajo = frac_radio_bajo * rmax

    low_mask = rr <= rbajo
    high_mask = rr > rbajo

    e_low = energia[low_mask].sum()
    e_high = energia[high_mask].sum()

    return float(e_high / (e_low + 1e-8))


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


def extraer_features_objeto(img_rgb, labels, lab, context_pad=12):
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


def etiquetar_componentes_por_semillas(
    labels,
    componentes,
    semillas_ameba,
    semillas_fondo=None,
    semillas_cristal=None,
    semillas_residuo=None,
):
    """
    Segunda red:
      1 = ameba
      0 = no_ameba

    Si una componente contiene:
      - semillas de ameba y no negativas -> 1
      - semillas negativas y no de ameba -> 0
      - ambas o ninguna -> se ignora
    """
    semillas_fondo = semillas_fondo or []
    semillas_cristal = semillas_cristal or []
    semillas_residuo = semillas_residuo or []

    semillas_neg = semillas_fondo + semillas_cristal + semillas_residuo

    labs_validos = []
    y_obj = []

    for lab in componentes:
        mask_lab = labels == lab

        pos = 0
        neg = 0

        for yy, xx in semillas_ameba:
            if (
                0 <= yy < labels.shape[0]
                and 0 <= xx < labels.shape[1]
                and mask_lab[yy, xx]
            ):
                pos += 1

        for yy, xx in semillas_neg:
            if (
                0 <= yy < labels.shape[0]
                and 0 <= xx < labels.shape[1]
                and mask_lab[yy, xx]
            ):
                neg += 1

        if pos > 0 and neg == 0:
            labs_validos.append(lab)
            y_obj.append(1)
        elif neg > 0 and pos == 0:
            labs_validos.append(lab)
            y_obj.append(0)

    return labs_validos, np.array(y_obj, dtype=np.int32)


def construir_dataset_segunda_red(
    img_rgb,
    mask_candidata,
    semillas_ameba,
    semillas_fondo=None,
    semillas_cristal=None,
    semillas_residuo=None,
    min_area=80,
):
    labels, stats, componentes = extraer_componentes(mask_candidata, min_area=min_area)

    labs_validos, y_obj = etiquetar_componentes_por_semillas(
        labels,
        componentes,
        semillas_ameba=semillas_ameba,
        semillas_fondo=semillas_fondo,
        semillas_cristal=semillas_cristal,
        semillas_residuo=semillas_residuo,
    )

    X_obj = []
    for lab in labs_validos:
        feat = extraer_features_objeto(img_rgb, labels, lab)
        X_obj.append(feat)

    if len(X_obj) == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            labels,
            [],
        )

    X_obj = np.vstack(X_obj).astype(np.float32)
    return X_obj, y_obj, labels, labs_validos


def normalizar_features_objeto(X, mu=None, sigma=None):
    X = X.astype(np.float32)

    if mu is None:
        mu = X.mean(axis=0)
    if sigma is None:
        sigma = X.std(axis=0)

    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    Xn = (X - mu) / sigma
    return Xn.astype(np.float32), mu.astype(np.float32), sigma.astype(np.float32)


def predict_labels(W, X, w2):
    X2 = np.sum(X * X, axis=1, keepdims=True)
    W2 = np.sum(W * W, axis=1)[None, :]
    dist2 = X2 + W2 - 2.0 * (X @ W.T)
    return w2[np.argmin(dist2, axis=1)]


def aplicar_segunda_red_por_objeto(
    img_rgb, mask_candidata, ruta_pkl="modelo_hlvq_objetos.pkl", min_area=80
):
    with open(ruta_pkl, "rb") as f:
        modelo2 = pickle.load(f)

    mu = modelo2["mu"]
    sigma = modelo2["sigma"]
    W = modelo2["W_best"]
    w2 = modelo2["w2"]

    labels, stats, componentes = extraer_componentes(mask_candidata, min_area=min_area)

    X_obj = []
    labs_validos = []

    for lab in componentes:
        feat = extraer_features_objeto(img_rgb, labels, lab)
        X_obj.append(feat)
        labs_validos.append(lab)

    if len(X_obj) == 0:
        return np.zeros_like(mask_candidata, dtype=np.uint8), labels, {}

    X_obj = np.vstack(X_obj).astype(np.float32)
    Xn, _, _ = normalizar_features_objeto(X_obj, mu=mu, sigma=sigma)

    pred = predict_labels(W, Xn, w2)

    mask_final = np.zeros_like(mask_candidata, dtype=np.uint8)
    pred_por_componente = {}

    for lab, cls in zip(labs_validos, pred):
        pred_por_componente[int(lab)] = int(cls)
        if cls == 1:
            mask_final[labels == lab] = 1

    return mask_final, labels, pred_por_componente


# =============================================================================
# SEGUNDA RED POR REGIONES
# =============================================================================


def obtener_centroides_componentes(labels, componentes):
    centroides = {}
    for lab in componentes:
        ys, xs = np.where(labels == lab)
        if len(ys) == 0:
            continue
        cy = int(np.mean(ys))
        cx = int(np.mean(xs))
        centroides[lab] = (cy, cx)
    return centroides


def mostrar_overlay_componentes(img_rgb, labels, componentes, etiquetas_dict=None):
    """
    etiquetas_dict:
      1 -> ameba
      0 -> no_ameba
      si no existe -> sin etiquetar
    """
    if etiquetas_dict is None:
        etiquetas_dict = {}

    overlay = img_rgb.copy()

    for lab in componentes:
        mask = labels == lab

        if etiquetas_dict.get(lab, None) == 1:
            color = np.array([255, 0, 0], dtype=np.float32)  # rojo
        elif etiquetas_dict.get(lab, None) == 0:
            color = np.array([0, 255, 255], dtype=np.float32)  # cian
        else:
            color = np.array([255, 255, 0], dtype=np.float32)  # amarillo

        overlay[mask] = (
            (0.55 * overlay[mask].astype(np.float32) + 0.45 * color)
            .clip(0, 255)
            .astype(np.uint8)
        )

    return overlay


def seleccionar_regiones_segunda_red(img_rgb, mask_candidata, min_area=80):
    """
    Controles:
      1 -> modo ameba
      0 -> modo no_ameba
      clic izquierdo -> etiqueta región
      u -> deshacer
      c -> limpiar
      g -> guardar y cerrar
    """
    labels, stats, componentes = extraer_componentes(mask_candidata, min_area=min_area)
    centroides = obtener_centroides_componentes(labels, componentes)

    etiquetas_dict = {}
    historial = []
    estado = {"modo": 1}

    fig, ax = plt.subplots(figsize=(9, 9))

    def refrescar():
        ax.clear()
        overlay = mostrar_overlay_componentes(
            img_rgb, labels, componentes, etiquetas_dict
        )
        ax.imshow(overlay)
        ax.axis("off")

        for lab, (cy, cx) in centroides.items():
            ax.text(
                cx,
                cy,
                str(lab),
                color="white",
                fontsize=8,
                ha="center",
                va="center",
                bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2"),
            )

        n_ameba = sum(1 for v in etiquetas_dict.values() if v == 1)
        n_no = sum(1 for v in etiquetas_dict.values() if v == 0)
        modo_txt = "AMEBA [1]" if estado["modo"] == 1 else "NO_AMEBA [0]"

        ax.set_title(
            f"Modo activo: {modo_txt} | Ameba: {n_ameba} | No_ameba: {n_no} | "
            "1=ameba, 0=no_ameba, clic=etiquetar, u=undo, c=clear, g=guardar"
        )
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        if event.button != 1:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))

        if not (0 <= x < labels.shape[1] and 0 <= y < labels.shape[0]):
            return

        lab = int(labels[y, x])
        if lab == 0 or lab not in componentes:
            return

        anterior = etiquetas_dict.get(lab, None)
        etiquetas_dict[lab] = estado["modo"]
        historial.append((lab, anterior))
        refrescar()

    def on_key(event):
        if event.key == "1":
            estado["modo"] = 1
        elif event.key == "0":
            estado["modo"] = 0
        elif event.key == "u":
            if historial:
                lab, anterior = historial.pop()
                if anterior is None:
                    etiquetas_dict.pop(lab, None)
                else:
                    etiquetas_dict[lab] = anterior
        elif event.key == "c":
            etiquetas_dict.clear()
            historial.clear()
        elif event.key == "g":
            plt.close(fig)
            return

        refrescar()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    refrescar()
    plt.tight_layout()
    plt.show(block=True)

    return labels, componentes, etiquetas_dict


def postprocesar_mascara_sin_semillas(
    mask_bin, kernel_size=5, min_area=50, close_iter=2, open_iter=2
):
    """Igual que antes — opera sobre una máscara binaria de UNA clase."""
    mask = (mask_bin > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    mask = rellenar_huecos(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if num_labels <= 1:
        return mask
    out = np.zeros_like(mask, dtype=np.uint8)
    for lab in range(1, num_labels):
        if stats[lab, cv2.CC_STAT_AREA] >= min_area:
            out[labels == lab] = 1
    return out


def construir_dataset_segunda_red_desde_regiones(img_rgb, labels, etiquetas_dict):
    """
    Devuelve:
      X_obj : (N, 3)
      y_obj : (N,)
      labs_validos : labels de componentes usados
    """
    X_obj = []
    y_obj = []
    labs_validos = []

    for lab, cls in etiquetas_dict.items():
        feat = extraer_features_objeto(img_rgb, labels, lab)
        X_obj.append(feat)
        y_obj.append(int(cls))
        labs_validos.append(int(lab))

    if len(X_obj) == 0:
        return (np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32), [])

    X_obj = np.vstack(X_obj).astype(np.float32)
    y_obj = np.array(y_obj, dtype=np.int32)

    return X_obj, y_obj, labs_validos


def split_dataset_objetos(X, y, val_frac=0.25, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)

    n_val = max(1, int(round(len(X) * val_frac)))
    n_val = min(len(X) - 1, n_val)

    idx_val = idx[:n_val]
    idx_tr = idx[n_val:]

    return X[idx_tr], y[idx_tr], X[idx_val], y[idx_val]


def visualizar_predicciones_segunda_red(img_rgb, labels, pred_por_componente):
    overlay = img_rgb.copy()
    componentes = list(pred_por_componente.keys())
    centroides = obtener_centroides_componentes(labels, componentes)

    for lab, cls in pred_por_componente.items():
        mask = labels == lab

        if cls == 1:
            color = np.array([255, 0, 0], dtype=np.float32)  # ameba
            txt = "A"
        else:
            color = np.array([0, 255, 255], dtype=np.float32)  # no_ameba
            txt = "N"

        overlay[mask] = (
            (0.55 * overlay[mask].astype(np.float32) + 0.45 * color)
            .clip(0, 255)
            .astype(np.uint8)
        )

        if lab in centroides:
            cy, cx = centroides[lab]
            cv2.putText(
                overlay,
                txt,
                (cx - 5, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    plt.figure()
    plt.imshow(overlay)
    plt.title("Predicción segunda red por región")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# =============================================================================
# Segmentación nueva
# =============================================================================

ruta = r"images1/amoeba_0009.jpg"

img_rgb = cv2.imread(ruta)

# Primera etapa: segmentación
mask_raw, labels_ws, _ = segmentar_ameba_completa(img_rgb)

# Filtrar labels por área mínima y reetiquetar
min_area = 80
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

# -------- Segunda red --------
mask_ameba_final_2, labels_obj, pred_por_componente = aplicar_segunda_red_por_objeto(
    img_rgb, labels, ruta_pkl="modelo_hlvq_SOMIB.pkl", min_area=min_area
)

with open("modelo_hlvq_SOMIB.pkl", "rb") as f:
    modelo = pickle.load(f)

print(modelo["features"])
print(modelo["mu"].shape)
print(modelo["sigma"].shape)
print(modelo["W_best"].shape)

labels_ameba = np.zeros_like(labels_obj, dtype=np.int32)

nuevo_id = 1
for lab, pred in pred_por_componente.items():
    if pred == 1:
        labels_ameba[labels_obj == lab] = nuevo_id
        nuevo_id += 1
plt.figure(figsize=(6, 6))
plt.imshow(labels_ameba, cmap="nipy_spectral")
plt.title("Amebas aceptadas por la HLVQ")
plt.axis("off")
plt.show()
# Visualización final
mascaras_finales = {
    "ameba": (mask_ameba_final_2, (220, 50, 50)),
}

mostrar_segmentacion_sin_semillas(img_rgb, mask_raw, mascaras_finales)

# Visualizar decisión por región
visualizar_predicciones_segunda_red(img_rgb, labels_obj, pred_por_componente)
