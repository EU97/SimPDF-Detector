# SimPDF-Detector

Detector de PDFs duplicados o similares por contenido textual y por similitud visual (primera página), pensado para carpetas locales.

## ¿Qué hace?

- Fase 1: Deduplica por hash (SHA-256) y detecta archivos 100% idénticos.
- Fase 2: Extrae para cada PDF su texto completo y una imagen de la primera página.
- Fase 3: Compara todos los pares restantes:
	- Similitud de texto con TF-IDF + coseno (con stopwords en español incluidas).
	- Similitud visual de la primera página con SSIM.
- Reporta pares similares por texto y por imagen (ordenados de mayor a menor similitud).

Limitación: La similitud visual solo usa la primera página, para acelerar el proceso.

## Requisitos

- Python 3.9 o superior (recomendado 3.10+)
- Windows, macOS o Linux. En Windows puede requerirse Microsoft C++ Build Tools para compilar/instalar algunas dependencias si no hay wheels precompilados.

Instala dependencias con `pip` usando el archivo `requirements.txt` incluido.

## Instalación (Windows PowerShell)

```powershell
# 1) Crear y activar un entorno virtual (opcional pero recomendado)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Instalar dependencias
pip install -r requirements.txt
```

Si tienes problemas con PyMuPDF o scikit-image, asegúrate de usar una versión reciente de Python y pip:

```powershell
python -m pip install --upgrade pip
```

## Configuración rápida

Edita las constantes al inicio de `src/simpdf.py` según tu necesidad:

- `TARGET_DIRECTORY`: Ruta de la carpeta donde están tus PDFs.
- `TEXT_SIMILARITY_THRESHOLD`: Umbral de similitud de texto (0.0 a 1.0). Por defecto 0.90.
- `VISUAL_SIMILARITY_THRESHOLD`: Umbral visual (0.0 a 1.0). Por defecto 0.95.
- `RENDER_DPI`: DPI para renderizar la primera página. Más alto = más lento y más preciso.
- `COMPARE_IMG_SIZE`: Tamaño al que se reescala la imagen para comparar.

El script incluye una lista estática de stopwords en español para mejorar la comparación textual y evitar errores de configuración en scikit-learn.

## Uso

```powershell
# Modo sencillo (usa valores por defecto del archivo)
python .\src\simpdf.py

# Modo con argumentos (recomendado)
python .\src\simpdf.py --dir "C:\\ruta\\a\\PDFs" --text-threshold 0.9 --visual-threshold 0.95 --dpi 96 --img-size 500 500 --visual-pages 3 --out-dir .\reports --include-dirs "proyectos,2025" --exclude-dirs "backup,temp"
```

El programa:

1) Recorre recursivamente `TARGET_DIRECTORY` buscando PDFs.
2) Reporta duplicados exactos (hash) en grupos.
3) Procesa los PDFs únicos restantes para extraer texto e imagen de las primeras N páginas (por defecto 1).
4) Compara todos los pares y muestra los que superen los umbrales configurados.

## Salida esperada (ejemplo)

- Grupos de duplicados exactos encontrados (si los hay).
- Lista de pares con alta similitud de texto (porcentaje).
- Lista de pares con alta similitud visual (porcentaje). Por defecto usa solo la primera página.
	- Si usas `--visual-pages N`, el valor mostrado es el promedio de SSIM de las primeras N páginas comunes entre ambos PDFs.

## Consejos y rendimiento

- Si tienes muchos PDFs, la Fase 3 puede tardar. Puedes bajar `RENDER_DPI` o usar un `COMPARE_IMG_SIZE` menor para acelerar.
- La similitud de texto se beneficia de PDFs con texto seleccionable. Si son escaneos sin OCR, la similitud textual será baja.
- Por rendimiento, por defecto se usa la primera página. Para analizar más páginas, usa `--visual-pages` para calcular el promedio.

## Argumentos disponibles

- `--dir`: Carpeta con PDFs a analizar. Por defecto, el valor de `TARGET_DIRECTORY` en el script.
- `--text-threshold`: Umbral de similitud de texto (0–1). Default: 0.90.
- `--visual-threshold`: Umbral de similitud visual (0–1). Default: 0.95.
- `--dpi`: DPI para renderizar cada página. Default: 96.
- `--img-size W H`: Tamaño de reescalado (ancho y alto) antes de comparar. Default: 500 500.
- `--visual-pages`: Número de páginas a considerar para la similitud visual (desde la primera). Se calcula el promedio SSIM. Default: 1.
- `--out-dir`: Carpeta donde se exportan los reportes CSV/JSON. Default: `./reports`.
- `--include-dirs`: Lista separada por comas de subcadenas; solo se analizan directorios cuyo path contenga alguna de estas.
- `--exclude-dirs`: Lista separada por comas de subcadenas; se descartan directorios cuyo path contenga alguna de estas.

## Reportes exportados

Se generan automáticamente en `--out-dir`:

- `duplicates.csv`: grupos de duplicados exactos (por hash).
- `text_similar.csv`: pares con similitud de texto por encima del umbral.
- `visual_similar.csv`: pares con similitud visual por encima del umbral.
- `summary.json`: resumen con configuración utilizada y resultados completos.
- `report.csv`: CSV consolidado con todos los resultados (tipo, grupo, archivos y similitud).

## Licencia

MIT (o la que prefieras).

## Solución de problemas

- Error de stopwords al usar scikit-learn: ya está resuelto con una lista interna de stopwords en español.
- Problemas al instalar dependencias en Windows: actualiza `pip`, usa Python 3.10+ y prueba desde un entorno virtual limpio.
- PyMuPDF (fitz) no abre algunos PDFs: confirma que los archivos no estén corruptos o protegidos.


