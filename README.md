# Amoeba Segmentation Morphology

Pipeline de procesamiento de imagen para segmentar amebas en micrografias. El proyecto combina filtros morfologicos, filtrado en frecuencia, watershed por islas y una segunda etapa de clasificacion por objeto con un modelo Hybrid LVQ.

## Vista rapida

| Imagen original | Imagen procesada |
| --- | --- |
| ![Ameba original](images1/amoeba_0001.jpg) | ![Ameba procesada](processed_images/amoeba_0001_procesada.jpg) |

## Contexto

Este repositorio esta preparado para subir a GitHub como material de apoyo para el congreso SOMIB. El codigo y los ejemplos documentan el flujo usado para segmentacion de amebas en imagenes de microscopia.

## Que incluye

- `Metodo_SOMIB.py`: pipeline principal de segmentacion, filtros adaptativos, diagnostico de imagen, mascara candidata y separacion con watershed.
- `etiquetar_regiones_carpeta.py`: interfaz interactiva para etiquetar regiones candidatas como `ameba` o `no_ameba`.
- `Segmentacion_con_filtro.py`: utilidades de entrenamiento, extraccion de caracteristicas por objeto, Hybrid LVQ y aplicacion de la segunda red.
- `images1/`: imagenes de ejemplo.
- `processed_images/`: resultados procesados de ejemplo.
- `modelo_hlvq_SOMIB.pkl`: modelo entrenado incluido como referencia para los experimentos preparados para SOMIB.

## Requisitos

- Python 3.10 o superior recomendado.
- Linux, macOS o Windows con soporte para ventanas graficas si se usara la herramienta interactiva de etiquetado.
- Dependencias listadas en `requirements.txt`.

Instalacion sugerida:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Uso rapido

### 1. Segmentar una imagen desde Python

```python
import cv2
from Metodo_SOMIB import segmentar_ameba_completa

img_bgr = cv2.imread("images1/amoeba_0001.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

mask_bin, labels_ws = segmentar_ameba_completa(img_rgb=img_rgb)
```

`mask_bin` contiene la mascara binaria candidata y `labels_ws` contiene los objetos separados por watershed.

### 2. Etiquetar regiones de una carpeta

```bash
python etiquetar_regiones_carpeta.py images1 etiquetas_regiones_SOMIB.pkl
```

Controles de la ventana interactiva:

- `1`: modo ameba.
- `0`: modo no ameba.
- clic izquierdo: etiqueta la region seleccionada.
- `u`: deshacer ultimo clic.
- `c`: limpiar etiquetas de la imagen actual.
- `g`: guardar todas las etiquetas y cerrar.

### 3. Cargar etiquetas y preparar dataset

```python
from etiquetar_regiones_carpeta import cargar_etiquetas, preparar_dataset_segunda_red

datos = cargar_etiquetas("etiquetas_regiones_SOMIB.pkl")
dataset = preparar_dataset_segunda_red(datos, carpeta="images1", val_frac=0.25, seed=0)
```

### 4. Aplicar una segunda red entrenada por objeto

```python
import cv2
from Metodo_SOMIB import segmentar_ameba_completa
from Segmentacion_con_filtro import aplicar_segunda_red_por_objeto

img_bgr = cv2.imread("images1/amoeba_0001.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

mask_candidata, _ = segmentar_ameba_completa(img_rgb=img_rgb)
mask_final, labels, predicciones = aplicar_segunda_red_por_objeto(
    img_rgb,
    mask_candidata,
    ruta_pkl="modelo_hlvq_SOMIB.pkl",
)
```

## Flujo de trabajo recomendado

1. Colocar imagenes de microscopia en una carpeta local.
2. Ejecutar `segmentar_ameba_completa` para generar regiones candidatas.
3. Usar `etiquetar_regiones_carpeta.py` para crear un archivo `.pkl` de etiquetas.
4. Preparar el dataset con `preparar_dataset_segunda_red`.
5. Entrenar o ajustar la segunda red con las funciones de `Segmentacion_con_filtro.py`.
6. Validar resultados visualmente y guardar ejemplos en `processed_images/`.

## Estructura del repositorio

```text
.
|-- Metodo_SOMIB.py
|-- Segmentacion_con_filtro.py
|-- etiquetar_regiones_carpeta.py
|-- images1/
|-- processed_images/
|-- modelo_hlvq_SOMIB.pkl
|-- requirements.txt
|-- LICENSE
`-- README.md
```

## Notas

- Los scripts usan `matplotlib` con backend Qt para visualizacion interactiva; en servidores sin interfaz grafica puede ser necesario cambiar el backend o ejecutar solo las partes no interactivas.
- Las anotaciones generadas (`etiquetas_regiones*.pkl`) se tratan como salidas locales y no se recomiendan para versionar salvo que formen parte del dataset final.
- `processed_images/` contiene ejemplos utiles para documentar resultados, pero los resultados nuevos pueden regenerarse a partir de `images1/` y los scripts.

## Licencia

Este proyecto esta publicado bajo licencia MIT. Consulta [LICENSE](LICENSE) para mas detalles.
