import matplotlib

matplotlib.use("Qt5Agg")
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import generic_filter
from skimage.segmentation import watershed


class NodoProcesamiento(ABC):
    """
    Cada nodo recibe una imagen y devuelve una imagen procesada.
    Todos los nodos comparten la misma interfaz para poderlos
    encadenar arbitrariamente.
    """

    nombre: str = "base"

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def aplicar(self, img: np.ndarray) -> np.ndarray: ...

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.aplicar(img)

    def __repr__(self):
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.nombre}({params_str})"


class FiltradoHomomorfico(NodoProcesamiento):
    """
    Separa la componente de iluminación (baja frecuencia) de la
    reflectancia (alta frecuencia) y atenúa la primera.
    Útil cuando rango_iluminacion es alto.
    """

    nombre = "homomorfico"

    def __init__(self, sigma=30, gain_low=0.3, gain_high=1.5):
        super().__init__(sigma=sigma, gain_low=gain_low, gain_high=gain_high)

    def aplicar(self, img):
        img_f = img.astype(np.float32) + 1.0
        log_img = np.log(img_f)
        log_ilum = cv2.GaussianBlur(log_img, (0, 0), sigmaX=self.params["sigma"])
        log_refl = log_img - log_ilum
        resultado = (
            self.params["gain_low"] * log_ilum + self.params["gain_high"] * log_refl
        )
        resultado = np.exp(resultado) - 1.0
        return cv2.normalize(resultado, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


class BackgroundSubtraction(NodoProcesamiento):
    """
    Estima el fondo con un blur gaussiano y lo resta.
    Más rápido que homomórfico pero menos sofisticado.
    """

    nombre = "background_subtraction"

    def __init__(self, sigma=50):
        super().__init__(sigma=sigma)

    def aplicar(self, img):
        fondo = cv2.GaussianBlur(img, (0, 0), sigmaX=self.params["sigma"])
        resultado = cv2.subtract(img, fondo)
        return cv2.normalize(resultado, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


class FiltroBilateral(NodoProcesamiento):
    """
    Filtro bilateral: suaviza preservando bordes.
    Para ruido moderado (sigma_ruido entre 5 y 10).
    """

    nombre = "bilateral"

    def __init__(self, d=5, sigmaColor=25, sigmaSpace=5):
        super().__init__(d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

    def aplicar(self, img):
        return cv2.bilateralFilter(
            img,
            d=self.params["d"],
            sigmaColor=self.params["sigmaColor"],
            sigmaSpace=self.params["sigmaSpace"],
        )


class NonLocalMeans(NodoProcesamiento):
    """
    Non-local means: el mejor denoising sin red neuronal.
    Más lento pero excelente para ruido alto (sigma_ruido > 11).
    """

    nombre = "non_local_means"

    def __init__(self, h=10, templateWindow=7, searchWindow=21):
        super().__init__(h=h, templateWindow=templateWindow, searchWindow=searchWindow)

    def aplicar(self, img):
        return cv2.fastNlMeansDenoising(
            img,
            h=self.params["h"],
            templateWindowSize=self.params["templateWindow"],
            searchWindowSize=self.params["searchWindow"],
        )


class FiltroGaussiano(NodoProcesamiento):
    """
    Suavizado simple. Útil como referencia o para limpieza muy ligera.
    """

    nombre = "gaussiano"

    def __init__(self, sigma=1.0):
        super().__init__(sigma=sigma)

    def aplicar(self, img):
        return cv2.GaussianBlur(img, (0, 0), sigmaX=self.params["sigma"])


class FiltroMediana(NodoProcesamiento):
    """
    Mediana: elimina ruido de sal y pimienta sin afectar bordes.
    """

    nombre = "mediana"

    def __init__(self, ksize=3):
        super().__init__(ksize=ksize)

    def aplicar(self, img):
        return cv2.medianBlur(img, self.params["ksize"])


class CLAHE(NodoProcesamiento):
    """
    Contrast Limited Adaptive Histogram Equalization.
    Realza contraste localmente sin saturar áreas planas.
    """

    nombre = "clahe"

    def __init__(self, clipLimit=2.5, tileGridSize=(8, 8)):
        super().__init__(clipLimit=clipLimit, tileGridSize=tileGridSize)

    def aplicar(self, img):
        clahe = cv2.createCLAHE(
            clipLimit=self.params["clipLimit"], tileGridSize=self.params["tileGridSize"]
        )
        return clahe.apply(img)


class ContrastStretch(NodoProcesamiento):
    """
    Estiramiento lineal de contraste por percentiles.
    Mapea [p_low, p_high] → [0, 255].
    """

    nombre = "contrast_stretch"

    def __init__(self, p_low=2, p_high=98):
        super().__init__(p_low=p_low, p_high=p_high)

    def aplicar(self, img):
        lo = np.percentile(img, self.params["p_low"])
        hi = np.percentile(img, self.params["p_high"])
        if hi <= lo:
            return img
        out = (img.astype(np.float32) - lo) * 255.0 / (hi - lo)
        return np.clip(out, 0, 255).astype(np.uint8)


class GammaCorrection(NodoProcesamiento):
    """
    Corrección gamma: gamma<1 aclara, gamma>1 oscurece.
    """

    nombre = "gamma"

    def __init__(self, gamma=1.0):
        super().__init__(gamma=gamma)

    def aplicar(self, img):
        inv = 1.0 / self.params["gamma"]
        tabla = ((np.arange(256) / 255.0) ** inv * 255).astype(np.uint8)
        return cv2.LUT(img, tabla)


class UnsharpMask(NodoProcesamiento):
    """
    Realza bordes restando una versión suavizada de la imagen.
    """

    nombre = "unsharp_mask"

    def __init__(self, strength=1.5, sigma=1.5):
        super().__init__(strength=strength, sigma=sigma)

    def aplicar(self, img):
        s = self.params["strength"]
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=self.params["sigma"])
        out = cv2.addWeighted(img, 1 + s, blurred, -s, 0)
        return np.clip(out, 0, 255).astype(np.uint8)


class HighBoost(NodoProcesamiento):
    """
    Filtrado high-boost: similar a unsharp pero con factor amplificador.
    out = A*original - blurred, con A>1
    """

    nombre = "high_boost"

    def __init__(self, A=1.5, sigma=2.0):
        super().__init__(A=A, sigma=sigma)

    def aplicar(self, img):
        A = self.params["A"]
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=self.params["sigma"])
        out = A * img.astype(np.float32) - blurred.astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)


class FiltroFrecuencia(NodoProcesamiento):
    """
    Filtro en dominio de frecuencia con varias formas posibles:
      - 'pasa_altas_ideal'
      - 'pasa_altas_butterworth'
      - 'pasa_altas_gaussiano'
      - 'pasa_bajas_ideal'
      - 'pasa_bajas_butterworth'
      - 'pasa_bajas_gaussiano'
      - 'pasa_banda'  (requiere fc_low y fc_high)
    """

    nombre = "filtro_frecuencia"

    def __init__(
        self, tipo="pasa_altas_gaussiano", fc=30, fc_low=10, fc_high=80, orden=2
    ):
        super().__init__(tipo=tipo, fc=fc, fc_low=fc_low, fc_high=fc_high, orden=orden)

    def _construir_filtro(self, shape):
        H, W = shape
        cy, cx = H // 2, W // 2
        Y, X = np.ogrid[:H, :W]
        D = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

        tipo = self.params["tipo"]
        fc = self.params["fc"]
        n = self.params["orden"]

        if tipo == "pasa_altas_ideal":
            return (D >= fc).astype(np.float32)
        if tipo == "pasa_bajas_ideal":
            return (D <= fc).astype(np.float32)
        if tipo == "pasa_altas_gaussiano":
            return 1 - np.exp(-(D**2) / (2 * fc**2))
        if tipo == "pasa_bajas_gaussiano":
            return np.exp(-(D**2) / (2 * fc**2))
        if tipo == "pasa_altas_butterworth":
            return 1.0 / (1.0 + (fc / (D + 1e-8)) ** (2 * n))
        if tipo == "pasa_bajas_butterworth":
            return 1.0 / (1.0 + (D / fc) ** (2 * n))
        if tipo == "pasa_banda":
            fl, fh = self.params["fc_low"], self.params["fc_high"]
            return ((D >= fl) & (D <= fh)).astype(np.float32)
        raise ValueError(f"tipo desconocido: {tipo}")

    def aplicar(self, img):
        F = np.fft.fftshift(np.fft.fft2(img.astype(np.float32)))
        H = self._construir_filtro(img.shape)
        F_filt = F * H
        out = np.fft.ifft2(np.fft.ifftshift(F_filt)).real
        return cv2.normalize(np.abs(out), None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )


class WaveletDenoise(NodoProcesamiento):
    """
    Denoising por umbralización en dominio wavelet.
    Preserva bordes y texturas mejor que un blur.
    """

    nombre = "wavelet_denoise"

    def __init__(self, wavelet="db4", level=2, mode="soft", sigma=None):
        super().__init__(wavelet=wavelet, level=level, mode=mode, sigma=sigma)

    def aplicar(self, img):
        import pywt

        coeffs = pywt.wavedec2(
            img.astype(np.float32),
            wavelet=self.params["wavelet"],
            level=self.params["level"],
        )
        # estimar sigma del ruido si no se dio
        sigma = self.params["sigma"]
        if sigma is None:
            # estimación robusta MAD (Donoho)
            cD = coeffs[-1][-1]  # detalles diagonales del último nivel
            sigma = np.median(np.abs(cD)) / 0.6745
        umbral = sigma * np.sqrt(2 * np.log(img.size))

        # umbralizar todos los coeficientes de detalle
        coeffs_umbralizados = [coeffs[0]]
        for nivel in coeffs[1:]:
            coeffs_umbralizados.append(
                tuple(
                    pywt.threshold(c, umbral, mode=self.params["mode"]) for c in nivel
                )
            )
        out = pywt.waverec2(coeffs_umbralizados, wavelet=self.params["wavelet"])
        out = out[: img.shape[0], : img.shape[1]]
        return np.clip(out, 0, 255).astype(np.uint8)


# =========================================================
# PIPELINE — encadena nodos
# =========================================================


class Pipeline:
    """
    Encadena varios nodos y los aplica en orden.
    Mantiene el historial de imágenes intermedias para visualización.
    """

    def __init__(self, nodos: list = None):
        self.nodos = nodos or []
        self.historial = []

    def agregar(self, nodo: NodoProcesamiento):
        self.nodos.append(nodo)
        return self

    def aplicar(self, img: np.ndarray, guardar_historial=True) -> np.ndarray:
        if guardar_historial:
            self.historial = [("original", img.copy())]
        actual = img
        for nodo in self.nodos:
            actual = nodo(actual)
            if guardar_historial:
                self.historial.append((str(nodo), actual.copy()))
        return actual

    def visualizar(self, cols=4):
        """Muestra todas las etapas del historial."""

        n = len(self.historial)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).flatten()
        for ax, (titulo, img) in zip(axes, self.historial):
            ax.imshow(img, cmap="gray")
            ax.set_title(titulo, fontsize=9)
            ax.axis("off")
        for ax in axes[n:]:
            ax.axis("off")
        plt.tight_layout()
        plt.show()


# =========================================================
# REGISTRO — para construir nodos desde diagnóstico
# =========================================================

REGISTRO_NODOS = {
    "filtrado_homomorfico": FiltradoHomomorfico,
    "background_subtraction": BackgroundSubtraction,
    "bilateral": FiltroBilateral,
    "non_local_means": NonLocalMeans,
    "gaussiano": FiltroGaussiano,
    "mediana": FiltroMediana,
    "clahe": CLAHE,
    "contrast_stretch": ContrastStretch,
    "gamma": GammaCorrection,
    "unsharp_mask": UnsharpMask,
    "high_boost": HighBoost,
    "filtro_frecuencia": FiltroFrecuencia,
    "wavelet_denoise": WaveletDenoise,
}


def construir_pipeline_desde_diagnostico(diag) -> Pipeline:
    """
    Toma un DiagnosticoImagen con pasos_recomendados ya generados
    y construye el Pipeline correspondiente.
    """
    pipeline = Pipeline()
    for paso in diag.pasos_recomendados:
        clase = REGISTRO_NODOS[paso["metodo"]]
        nodo = clase(**paso["params"])
        pipeline.agregar(nodo)
    return pipeline


# =========================================================
# NODO 1: ANÁLISIS — extrae métricas relevantes
# =========================================================


@dataclass
class DiagnosticoImagen:
    """
    Resultado del análisis de una imagen. Contiene métricas crudas y
    flags interpretables que el nodo de decisión usará.
    """

    # métricas crudas
    alto: int
    ancho: int
    intensidad_min: int
    intensidad_max: int
    rango_usado: int
    media: float
    std: float
    rango_iluminacion: float
    sigma_ruido: float
    nitidez: float
    gradiente_medio: float
    entropia: float

    # flags interpretables (los pone el nodo de decisión)
    iluminacion_desigual: bool = False
    es_ruidosa: bool = False
    es_borrosa: bool = False
    contraste_pobre: bool = False
    rango_comprimido: bool = False

    # decisiones (lista de pasos a aplicar, en orden)
    pasos_recomendados: List[Dict] = field(default_factory=list)

    def resumen(self) -> str:
        flags = []
        if self.iluminacion_desigual:
            flags.append("iluminación desigual")
        if self.es_ruidosa:
            flags.append("ruidosa")
        if self.es_borrosa:
            flags.append("borrosa")
        if self.contraste_pobre:
            flags.append("contraste pobre")
        if self.rango_comprimido:
            flags.append("rango comprimido")
        estado = ", ".join(flags) if flags else "sin problemas detectados"

        pasos = (
            " → ".join(p["nombre"] for p in self.pasos_recomendados)
            if self.pasos_recomendados
            else "ninguno"
        )

        return (
            f"Diagnóstico: {estado}\n"
            f"  rango_iluminacion = {self.rango_iluminacion:.1f}\n"
            f"  sigma_ruido       = {self.sigma_ruido:.2f}\n"
            f"  nitidez           = {self.nitidez:.0f}\n"
            f"  rango_usado       = {self.rango_usado}/255\n"
            f"  std               = {self.std:.2f}\n"
            f"Pipeline recomendado: {pasos}"
        )


def analizar_imagen(img: np.ndarray) -> DiagnosticoImagen:
    """
    NODO 1 — Extrae métricas crudas de la imagen, sin tomar decisiones.
    Solo mide.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    H, W = img.shape

    # iluminación de fondo
    sigma_ilum = max(W, H) // 8
    iluminacion = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma_ilum)
    rango_iluminacion = float(iluminacion.max() - iluminacion.min())

    # ruido
    img_suave = cv2.medianBlur(img, 5)
    sigma_ruido = float((img.astype(np.float32) - img_suave).std())

    # nitidez
    lap = cv2.Laplacian(img, cv2.CV_32F)
    nitidez = float(lap.var())

    # gradiente medio
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    gradiente_medio = float(np.sqrt(gx**2 + gy**2).mean())

    # entropía
    hist = np.bincount(img.flatten(), minlength=256).astype(np.float32)
    p = hist / hist.sum()
    p = p[p > 0]
    entropia = float(-np.sum(p * np.log2(p)))

    return DiagnosticoImagen(
        alto=H,
        ancho=W,
        intensidad_min=int(img.min()),
        intensidad_max=int(img.max()),
        rango_usado=int(img.max() - img.min()),
        media=float(img.mean()),
        std=float(img.std()),
        rango_iluminacion=rango_iluminacion,
        sigma_ruido=sigma_ruido,
        nitidez=nitidez,
        gradiente_medio=gradiente_medio,
        entropia=entropia,
    )


def procesar_todas_las_islas(
    img_filtrada, mask_bin, padding=8, area_min=20, k_small=5, k_large=13
):
    """
    Encuentra todas las islas y procesa cada una por separado.

    Retorna
    -------
    resultados_islas : list[dict]
    mapas_globales : dict[str, np.ndarray]
    labels : np.ndarray
    """

    mask_bin = (mask_bin > 0).astype(np.uint8)

    num_labels, labels, stats_cc, _ = cv2.connectedComponentsWithStats(
        mask_bin, connectivity=8
    )

    H, W = img_filtrada.shape[:2]

    nombres_mapas = [
        "entropia_13",
    ]

    mapas_globales = {
        nombre: np.zeros((H, W), dtype=np.float32) for nombre in nombres_mapas
    }
    resultados_islas = []

    for label_id in range(1, num_labels):  # saltar fondo = 0
        area = stats_cc[label_id, cv2.CC_STAT_AREA]
        if area < area_min:
            continue

        res = procesar_isla(
            img_filtrada=img_filtrada,
            labels=labels,
            label_id=label_id,
            padding=padding,
            k_small=k_small,
            k_large=k_large,
        )

        if res is None:
            continue

        resultados_islas.append(res)

        x0, y0, x1, y1 = res["bbox_global"]
        mask_local = res["mask_local"] > 0

        mapa_local = res["mapas_locales"]["entropia_13"]

        roi = mapas_globales["entropia_13"][y0:y1, x0:x1]
        roi[mask_local] = mapa_local[mask_local]
        mapas_globales["entropia_13"][y0:y1, x0:x1] = roi

    return resultados_islas, mapas_globales, labels


def procesar_isla(img_filtrada, labels, label_id, padding=8, k_small=5, k_large=13):
    """
    Procesa una sola isla (componente conectada) usando la imagen filtrada
    como fuente de mediciones, con un parche local y padding.

    Parámetros
    ----------
    img_filtrada : np.ndarray
        Imagen en gris ya filtrada.
    labels : np.ndarray
        Imagen de etiquetas de connected components.
    label_id : int
        Etiqueta de la isla a procesar.
    padding : int
        Margen alrededor del bounding box.
    k_small : int
        Kernel pequeño.
    k_large : int
        Kernel grande.

    Retorna
    -------
    resultado : dict
        Contiene máscara local, bbox, mapas locales y estadísticas resumen.
    """

    # máscara global de la isla
    mask_global = (labels == label_id).astype(np.uint8)

    ys, xs = np.where(mask_global > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x = xs.min()
    y = ys.min()
    w = xs.max() - xs.min() + 1
    h = ys.max() - ys.min() + 1

    x0, y0, x1, y1 = extraer_bbox_con_padding(
        x, y, w, h, img_filtrada.shape, padding=padding
    )

    # parche local real
    patch_img = img_filtrada[y0:y1, x0:x1].copy()
    patch_mask = mask_global[y0:y1, x0:x1].copy().astype(np.uint8)

    # versión uint8
    patch_img_u8 = patch_img.astype(np.uint8)
    patch_mask_u8 = (patch_mask > 0).astype(np.uint8) * 255

    ent_13 = calcular_mapa_entropia(patch_img_u8, ksize=k_large)

    # dejar solo la isla
    mask_bool = patch_mask > 0
    mapas = {
        "entropia_13": ent_13,
    }

    # copiar para no tocar originales
    mapas_isla = {}
    for nombre, mapa in mapas.items():
        tmp = mapa.astype(np.float32).copy()
        tmp[~mask_bool] = 0.0
        mapas_isla[nombre] = tmp

    resultado = {
        "label_id": label_id,
        "bbox_global": (x0, y0, x1, y1),
        "mask_local": patch_mask,
        "patch_img": patch_img_u8,
        "mapas_locales": mapas_isla,
    }

    return resultado


def extraer_bbox_con_padding(x, y, w, h, shape, padding=8):
    H, W = shape[:2]
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(W, x + w + padding)
    y1 = min(H, y + h + padding)
    return x0, y0, x1, y1


def calcular_mapa_entropia(img_gray, ksize=13):
    """
    Calcula entropía local usando una ventana ksize x ksize.
    """
    return generic_filter(img_gray, entropia_local, size=ksize)


def entropia_local(window):
    """
    Calcula la entropía de una ventana local.
    La ventana llega aplanada.
    """
    window = window.astype(np.uint8)
    hist = np.bincount(window, minlength=256).astype(np.float32)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def mascara_dos_maximos(entropia_map):
    # obtener valores únicos ordenados
    valores_unicos = np.unique(entropia_map)

    # tomar los dos más grandes
    top2 = valores_unicos[-2:]

    # crear máscara
    mask = np.isin(entropia_map, top2).astype(np.uint8)

    return mask


UMBRALES_DEFAULT = {
    "rango_iluminacion_alto": 61.03,  # mediana=16, p75=16, outliers > 30
    "sigma_ruido_alto": 11.46,  # mediana=7.3, p75=8.0
    "sigma_ruido_muy_alto": 8.09,  # cola del histograma
    "nitidez_baja": 201.22,  # mediana=364, p25=278
    "nitidez_muy_baja": 190.51,  # casos extremos
    "std_baja": 5.01,  # std mediana=13.7
    "rango_comprimido": 168.97,  # rango_usado mediana=245
}


def decidir_pipeline(
    diag: DiagnosticoImagen,
    img: np.ndarray,
    umbrales: Dict = None,
    guardar_historial: bool = True,
) -> tuple[np.ndarray, DiagnosticoImagen, Pipeline]:
    """
    NODO 2 — Recibe un diagnóstico y la imagen original.
    Agrega flags, decide qué pasos aplicar, construye el pipeline,
    lo ejecuta y devuelve (imagen_procesada, diagnóstico, pipeline).

    El pipeline retornado conserva el historial para visualización.
    """
    if umbrales is None:
        umbrales = UMBRALES_DEFAULT

    # ── flags ──────────────────────────────────────────────────
    diag.iluminacion_desigual = (
        diag.rango_iluminacion > umbrales["rango_iluminacion_alto"]
    )
    diag.es_ruidosa = diag.sigma_ruido > umbrales["sigma_ruido_alto"]
    diag.es_borrosa = diag.nitidez < umbrales["nitidez_baja"]
    diag.contraste_pobre = diag.std < umbrales["std_baja"]
    diag.rango_comprimido = diag.rango_usado < umbrales["rango_comprimido"]

    pasos = []

    # ── ORDEN: iluminación → ruido → contraste → bordes ────────

    # 1. corregir iluminación si es desigual
    if diag.iluminacion_desigual:
        pasos.append(
            {
                "nombre": "homomorfico",
                "metodo": "filtrado_homomorfico",
                "params": {
                    "sigma": min(diag.alto, diag.ancho) // 8,
                    "gain_low": 0.3,
                    "gain_high": 1.5,
                },
            }
        )

    # 2. denoising según nivel de ruido
    if diag.sigma_ruido > umbrales["sigma_ruido_muy_alto"]:
        pasos.append(
            {
                "nombre": "denoise_nlm",
                "metodo": "non_local_means",
                "params": {"h": 10, "templateWindow": 7, "searchWindow": 21},
            }
        )
    elif diag.es_ruidosa:
        pasos.append(
            {
                "nombre": "denoise_bilateral",
                "metodo": "bilateral",
                "params": {"d": 5, "sigmaColor": 25, "sigmaSpace": 5},
            }
        )

    # 3. realzar contraste si está pobre o el rango está comprimido
    if diag.contraste_pobre or diag.rango_comprimido:
        pasos.append(
            {
                "nombre": "clahe",
                "metodo": "clahe",
                "params": {"clipLimit": 2.5, "tileGridSize": (8, 8)},
            }
        )

    # 4. realzar bordes si la imagen es borrosa
    if diag.nitidez < umbrales["nitidez_muy_baja"]:
        pasos.append(
            {
                "nombre": "unsharp_fuerte",
                "metodo": "unsharp_mask",
                "params": {"strength": 2.5, "sigma": 1.5},
            }
        )
    elif diag.es_borrosa:
        pasos.append(
            {
                "nombre": "unsharp_suave",
                "metodo": "unsharp_mask",
                "params": {"strength": 1.2, "sigma": 1.5},
            }
        )

    diag.pasos_recomendados = pasos

    # ── construir y ejecutar pipeline ──────────────────────────
    pipeline = construir_pipeline_desde_diagnostico(diag)
    img_procesada = pipeline.aplicar(img, guardar_historial=guardar_historial)

    return img_procesada, diag, pipeline


def watershed_por_islas(mask_bin, imagen_gris, dist_thresh=0.35, min_area_isla=50):
    """
    Aplica Watershed de forma local (isla por isla) para evitar que
    las amebas pequeñas sean ignoradas por el máximo de distancia global.
    """
    # 1. Obtenemos las islas globales separadas
    num_islas, labels_islas, stats_islas, _ = cv2.connectedComponentsWithStats(
        mask_bin.astype(np.uint8), connectivity=4
    )

    # Creamos el lienzo negro donde iremos pegando los segmentos resultantes
    labels_finales = np.zeros_like(mask_bin, dtype=np.int32)
    max_id_global = 0

    # 2. Iteramos sobre cada isla (saltando el 0 que es el fondo)
    for lab in range(1, num_islas):
        area_isla = stats_islas[lab, cv2.CC_STAT_AREA]
        if area_isla < min_area_isla:  # Ignorar basurita minúscula
            continue

        # Extraer las coordenadas del Bounding Box de la isla
        x = stats_islas[lab, cv2.CC_STAT_LEFT]
        y = stats_islas[lab, cv2.CC_STAT_TOP]
        w = stats_islas[lab, cv2.CC_STAT_WIDTH]
        h = stats_islas[lab, cv2.CC_STAT_HEIGHT]

        # 3. Recortar el parche de la imagen y su máscara local
        roi_img = imagen_gris[y : y + h, x : x + w]
        roi_mask = (labels_islas[y : y + h, x : x + w] == lab).astype(np.uint8)

        # ====================================================
        # APLICAR WATERSHED LOCALMENTE AL PARCHE
        # ====================================================
        # Transformada de distancia local
        dist = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)

        # ¡LA MAGIA OCURRE AQUÍ! dist.max() ahora es exclusivo de esta ameba
        sure_fg = (dist > dist_thresh * dist.max()).astype(np.uint8)

        # Fondo seguro local
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(roi_mask, kernel, iterations=1)

        # Región desconocida local
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marcadores locales
        num_markers, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 1] = 0

        roi_bgr = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)
        markers_locales = cv2.watershed(roi_bgr, markers)
        # ====================================================

        # 4. Mapear los IDs locales al lienzo global
        # En OpenCV Watershed: el fondo queda con 1, los bordes con -1, y los objetos > 1
        vista_global = labels_finales[y : y + h, x : x + w]
        segmentos_unicos = np.unique(markers_locales)

        for id_local in segmentos_unicos:
            # Ignoramos el fondo del parche (1) y las líneas de corte del watershed (-1)
            if id_local <= 1:
                continue

            max_id_global += 1

            # Encontramos los píxeles que pertenecen a este objeto DENTRO de la máscara
            zona_segmento = (roi_mask > 0) & (markers_locales == id_local)

            # Asignamos el ID global
            vista_global[zona_segmento] = max_id_global

    print(f"Watershed (Por Isla): Se detectaron {max_id_global} segmentos en total.")
    return labels_finales


def segmentar_ameba_completa(
    img_rgb=None,
    ruta_imagen=None,
    fc_2=90,
    padding=8,
    area_min=20,
    k_small=5,
    k_large=13,
    dist_thresh=0.5,
    min_area_isla=50,
    usar_imagen_filtrada_en_watershed=True,
    return_debug=False,
):
    """
    Segmentación SOMIB completa.

    Entrada
    -------
    img_rgb : np.ndarray, opcional
        Imagen RGB. Es la forma recomendada para usarla desde:
        - etiquetar_regiones_carpeta.py
        - Segmentacion_con_filtro.py

    ruta_imagen : str, opcional
        Ruta de imagen. Útil para pruebas directas desde Metodo_SOMIB.py.

    Parámetros principales
    ----------------------
    fc_2 : int
        Radio de corte del filtro pasa-altas en frecuencia.

    padding : int
        Padding usado al procesar islas.

    area_min : int
        Área mínima para procesar islas candidatas en la etapa de entropía.

    k_small : int
        Kernel pequeño reservado para compatibilidad.

    k_large : int
        Kernel grande usado en el mapa de entropía local.

    dist_thresh : float
        Umbral relativo para watershed por isla.

    min_area_isla : int
        Área mínima de isla para watershed.

    usar_imagen_filtrada_en_watershed : bool
        Si True, watershed usa img_final = img_procesada * mask_bin.
        Si False, watershed usa la imagen gris original.

    return_debug : bool
        Si True, devuelve también un diccionario con imágenes intermedias.

    Retorna
    -------
    mask_bin : np.ndarray uint8
        Máscara binaria candidata, valores 0/1.

    labels_ws : np.ndarray int32
        Labels finales separados por watershed.
        Fondo = 0, objetos = 1, 2, 3, ...

    debug : dict, opcional
        Solo si return_debug=True.
    """

    # =========================================================
    # 1) Cargar / validar imagen
    # =========================================================
    if img_rgb is None and ruta_imagen is None:
        raise ValueError("Debes proporcionar img_rgb o ruta_imagen.")

    if img_rgb is None:
        imagen_col = cv2.imread(str(ruta_imagen))
        if imagen_col is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {ruta_imagen}")

        # cv2 lee BGR; convertimos a RGB para mantener compatibilidad.
        img_rgb = cv2.cvtColor(imagen_col, cv2.COLOR_BGR2RGB)

    if img_rgb is None:
        raise ValueError("No se recibió una imagen válida.")

    if img_rgb.ndim == 2:
        imagen = img_rgb.astype(np.uint8)
    elif img_rgb.ndim == 3:
        # Tus otros scripts pasan img_rgb, por eso usamos RGB2GRAY.
        imagen = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"Formato de imagen no soportado: shape={img_rgb.shape}")

    imagen = imagen.astype(np.uint8)

    # =========================================================
    # 2) Filtro pasa-altas en frecuencia
    # =========================================================
    image_fft = np.fft.fftshift(np.fft.fft2(imagen))

    rows, cols = imagen.shape
    center_row, center_col = rows // 2, cols // 2

    Y, X = np.ogrid[:rows, :cols]
    distancia_al_centro = np.sqrt((X - center_col) ** 2 + (Y - center_row) ** 2)

    filtro_hp = (distancia_al_centro >= fc_2).astype(np.float32)

    convolucion_hp = filtro_hp * image_fft
    real_hp = np.fft.ifft2(np.fft.ifftshift(convolucion_hp))
    img_new_hp = np.abs(real_hp)

    img_normalizada = cv2.normalize(img_new_hp, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )

    # =========================================================
    # 3) Otsu + cierre morfológico
    # =========================================================
    _, mask_o = cv2.threshold(
        img_normalizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)

    mask_cerrada = cv2.dilate(mask_o, kernel, iterations=2)
    mask_cerrada = cv2.morphologyEx(mask_cerrada, cv2.MORPH_CLOSE, kernel, iterations=1)

    # =========================================================
    # 4) Procesamiento por islas + mapa de entropía
    # =========================================================
    resultados_islas, mapas_globales, labels_iniciales = procesar_todas_las_islas(
        img_filtrada=img_normalizada,
        mask_bin=mask_cerrada,
        padding=padding,
        area_min=area_min,
        k_small=k_small,
        k_large=k_large,
    )

    ent_map = mapas_globales["entropia_13"]

    # Máscara por los dos máximos de entropía.
    # Se fuerza ent_map > 0 para evitar que el fondo entre como candidato
    # cuando el mapa tenga muchos ceros.
    mask_bin = mascara_dos_maximos(ent_map)
    mask_bin = ((mask_bin > 0) & (ent_map > 0)).astype(np.uint8)

    # Si no hay candidatos, devolver vacío sin romper el pipeline.
    if np.sum(mask_bin) == 0:
        labels_ws = np.zeros_like(mask_bin, dtype=np.int32)

        if return_debug:
            debug = {
                "imagen_gray": imagen,
                "image_fft": image_fft,
                "filtro_hp": filtro_hp,
                "img_new_hp": img_new_hp,
                "img_normalizada": img_normalizada,
                "mask_o": mask_o,
                "mask_cerrada": mask_cerrada,
                "ent_map": ent_map,
                "labels_iniciales": labels_iniciales,
                "resultados_islas": resultados_islas,
                "img_procesada": imagen,
                "img_final": imagen * mask_bin,
            }
            return mask_bin, labels_ws, debug

        return mask_bin, labels_ws

    # =========================================================
    # 5) Diagnóstico + filtrado adaptativo
    # =========================================================
    diag = analizar_imagen(imagen)
    img_procesada, diag, _ = decidir_pipeline(
        diag,
        imagen,
        guardar_historial=False,
    )

    if usar_imagen_filtrada_en_watershed:
        img_final = img_procesada.astype(np.float32) * mask_bin.astype(np.float32)
        img_final = cv2.normalize(img_final, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        imagen_watershed = img_final
    else:
        img_final = imagen.astype(np.float32) * mask_bin.astype(np.float32)
        img_final = cv2.normalize(img_final, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        imagen_watershed = imagen

    # =========================================================
    # 6) Watershed por islas
    # =========================================================
    labels_ws = watershed_por_islas(
        mask_bin=mask_bin,
        imagen_gris=imagen_watershed,
        dist_thresh=dist_thresh,
        min_area_isla=min_area_isla,
    )

    labels_ws = labels_ws.astype(np.int32)

    if return_debug:
        debug = {
            "imagen_gray": imagen,
            "image_fft": image_fft,
            "filtro_hp": filtro_hp,
            "img_new_hp": img_new_hp,
            "img_normalizada": img_normalizada,
            "mask_o": mask_o,
            "mask_cerrada": mask_cerrada,
            "ent_map": ent_map,
            "labels_iniciales": labels_iniciales,
            "resultados_islas": resultados_islas,
            "diag": diag,
            "img_procesada": img_procesada,
            "img_final": img_final,
            "imagen_watershed": imagen_watershed,
        }
        return mask_bin, labels_ws, debug

    return mask_bin, labels_ws, img_procesada
