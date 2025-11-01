import os
import argparse
import csv
import json
import hashlib
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from skimage.color import rgb2gray

# --- CONFIGURACIÓN PRINCIPAL ---

# ⬇️ IMPORTANTE: Cambia esta ruta a tu carpeta principal
TARGET_DIRECTORY = r"C:\Users\edgar\Desktop\PDF"

# ⬇️ PUEDES AJUSTAR ESTOS UMBRALES
# Umbral para similitud de TEXTO (0.0 a 1.0)
TEXT_SIMILARITY_THRESHOLD = 0.90  # 90%

# Umbral para similitud VISUAL (0.0 a 1.0)
VISUAL_SIMILARITY_THRESHOLD = 0.95 # 95% (la similitud visual suele ser más alta)

# Resolución (DPI) para renderizar la imagen. Más alto = más lento pero más preciso.
RENDER_DPI = 96 
# Tamaño estándar para comparar imágenes. Más grande = más lento.
COMPARE_IMG_SIZE = (500, 500)
# Número de páginas a usar para la similitud visual (desde la primera). 1 = solo portada.
MAX_VISUAL_PAGES = 1

# --- FIN DE LA CONFIGURACIÓN ---

# Lista estática de stopwords en español para TF-IDF
# Nota: scikit-learn no provee 'spanish' como stop list integrada.
# Usamos una lista común para evitar errores y mejorar resultados en español.
SPANISH_STOPWORDS = {
    'de','la','que','el','en','y','a','los','del','se','las','por','un','para','con','no','una','su','al','lo','como','más','pero','sus','le','ya','o','este','sí','porque','esta','entre','cuando','muy','sin','sobre','también','me','hasta','hay','donde','quien','desde','todo','nos','durante','todos','uno','les','ni','contra','otros','ese','eso','ante','ellos','e','esto','mí','antes','algunos','qué','unos','yo','otro','otras','otra','él','tanto','esa','estos','mucho','quienes','nada','muchos','cual','poco','ella','estar','estas','algunas','algo','nosotros','mi','mis','tú','te','ti','tu','tus','ellas','nosotras','vosotros','vosotras','os','mío','mía','míos','mías','tuyo','tuya','tuyos','tuyas','suyo','suya','suyos','suyas','nuestro','nuestra','nuestros','nuestras','vuestro','vuestra','vuestros','vuestras','esos','esas','estoy','estás','está','estamos','estáis','están','esté','estés','estemos','estéis','estén','estaré','estarás','estará','estaremos','estaréis','estarán','estaría','estarías','estaríamos','estaríais','estarían','estaba','estabas','estábamos','estabais','estaban','estuve','estuviste','estuvo','estuvimos','estuvisteis','estuvieron','estuviera','estuvieras','estuviéramos','estuvierais','estuvieran','estuviese','estuvieses','estuviésemos','estuvieseis','estuviesen','estando','estado','estada','estados','estadas','estad','he','has','ha','hemos','habéis','han','haya','hayas','hayamos','hayáis','hayan','habré','habrás','habrá','habremos','habréis','habrán','habría','habrías','habríamos','habríais','habrían','había','habías','habíamos','habíais','habían','hube','hubiste','hubo','hubimos','hubisteis','hubieron','hubiera','hubieras','hubiéramos','hubierais','hubieran','hubiese','hubieses','hubiésemos','hubieseis','hubiesen','habiendo','habido','habida','habidos','habidas','soy','eres','es','somos','sois','son','sea','seas','seamos','seáis','sean','seré','serás','será','seremos','seréis','serán','sería','serías','seríamos','seríais','serían','era','eras','éramos','erais','eran','fui','fuiste','fue','fuimos','fuisteis','fueron','fuera','fueras','fuéramos','fuerais','fueran','fuese','fueses','fuésemos','fueseis','fuesen','siendo','sido','tengo','tienes','tiene','tenemos','tenéis','tienen','tenga','tengas','tengamos','tengáis','tengan','tendré','tendrás','tendrá','tendremos','tendréis','tendrán','tendría','tendrías','tendríamos','tendríais','tendrían','tenía','tenías','teníamos','teníais','tenían','tuve','tuviste','tuvo','tuvimos','tuvisteis','tuvieron','tuviera','tuvieras','tuviéramos','tuvierais','tuvieran','tuviese','tuvieses','tuviésemos','tuvieseis','tuviesen','teniendo','tenido','tenida','tenidos','tenidas','tened'
}


def get_file_hash(filepath, block_size=65536):
    """Calcula el hash SHA-256 de un archivo."""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            while True:
                data = f.read(block_size)
                if not data:
                    break
                sha256.update(data)
        return sha256.hexdigest()
    except IOError as e:
        print(f"  [Error Hash] No se pudo leer {filepath}: {e}")
        return None

def preprocess_pdf(filepath, render_dpi=RENDER_DPI, compare_img_size=COMPARE_IMG_SIZE, max_pages=MAX_VISUAL_PAGES):
    """
    Extrae el texto completo y hasta 'max_pages' páginas como imágenes (escala de grises, reescaladas).
    Devuelve: (texto, [img_p1, img_p2, ...]) donde cada img es un np.array 2D normalizado [0,1].
    """
    pdf_text = ""
    images = []

    try:
        with fitz.open(filepath) as doc:
            # 1. Extraer Texto
            for page in doc:
                pdf_text += page.get_text()

            # 2. Extraer Imágenes de las primeras N páginas
            pages_to_process = min(max_pages, len(doc))
            for p in range(pages_to_process):
                page = doc.load_page(p)
                pix = page.get_pixmap(dpi=render_dpi)

                # Convertir a imagen PIL
                pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Convertir a array numpy
                np_img = np.array(pil_img)

                # Escala de grises para SSIM y reescalado
                gray_img = rgb2gray(np_img)
                std_img = resize(gray_img, compare_img_size, anti_aliasing=True)
                images.append(std_img)

    except Exception as e:
        print(f"  [Error Proceso] No se pudo procesar {filepath}: {e}")
        return "", []  # Devuelve datos vacíos si falla

    return pdf_text, images

def compare_images(imgs1, imgs2):
    """Compara dos listas de imágenes (páginas) usando SSIM promedio en las páginas comunes."""
    if not imgs1 or not imgs2:
        return 0.0

    k = min(len(imgs1), len(imgs2))
    if k == 0:
        return 0.0

    scores = []
    for i in range(k):
        a = imgs1[i]
        b = imgs2[i]
        # data_range es la diferencia entre max y min (1.0 - 0.0 = 1.0)
        try:
            score = ssim(a, b, data_range=1.0)
        except Exception:
            score = 0.0
        scores.append(score)

    return float(np.mean(scores)) if scores else 0.0

def compare_texts(text1, text2, vectorizer):
    """Compara dos textos usando TF-IDF y Similitud Coseno."""
    if not text1 or not text2:
        return 0.0
        
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except ValueError:
        # Puede ocurrir si el texto solo contiene "stop words"
        return 0.0

def find_all_similar_pdfs(
    directory,
    text_threshold=TEXT_SIMILARITY_THRESHOLD,
    visual_threshold=VISUAL_SIMILARITY_THRESHOLD,
    render_dpi=RENDER_DPI,
    compare_img_size=COMPARE_IMG_SIZE,
    max_visual_pages=MAX_VISUAL_PAGES,
    out_dir=None,
    include_dirs=None,
    exclude_dirs=None,
):
    
    pdf_files = []
    hashes = {}
    
    print(f"Buscando PDFs en: {directory}...")
    # Normalizar filtros a minúsculas para comparación case-insensitive
    include_dirs = [s.lower() for s in include_dirs] if include_dirs else []
    exclude_dirs = [s.lower() for s in exclude_dirs] if exclude_dirs else []

    for dirpath, _, filenames in os.walk(directory):
        norm_dirpath = dirpath.lower()

        # Excluir directorios cuyo path contenga alguno de los tokens excluidos
        if exclude_dirs and any(tok in norm_dirpath for tok in exclude_dirs):
            continue
        # Incluir solo directorios que coincidan con alguno de los tokens incluidos (si se proporcionan)
        if include_dirs and not any(tok in norm_dirpath for tok in include_dirs):
            continue
        for filename in filenames:
            if filename.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(dirpath, filename))
    
    if len(pdf_files) < 2:
        print("No se encontraron suficientes PDFs para comparar.")
        return
        
    print(f"Se encontraron {len(pdf_files)} archivos PDF.\n")

    # --- FASE 1: Buscar duplicados exactos (Hash) ---
    print("--- FASE 1: Buscando duplicados 100% exactos (por Hash) ---")
    unique_files = [] # Lista de archivos que no son duplicados exactos
    duplicate_groups = []
    
    for filepath in pdf_files:
        file_hash = get_file_hash(filepath)
        if file_hash is None:
            continue
            
        if file_hash in hashes:
            hashes[file_hash].append(filepath)
        else:
            hashes[file_hash] = [filepath]
    
    # Separa los únicos de los duplicados
    for file_hash, filepaths in hashes.items():
        if len(filepaths) > 1:
            duplicate_groups.append(filepaths)
        unique_files.append(filepaths[0]) # Añade solo UN representante de cada grupo

    # Reportar Fase 1
    if not duplicate_groups:
        print("✅ No se encontraron duplicados exactos.\n")
    else:
        print(f"🚨 Se encontraron {len(duplicate_groups)} grupos de duplicados exactos:\n")
        for i, group in enumerate(duplicate_groups):
            print(f"  Grupo {i+1}:")
            for path in group:
                print(f"    -> {path}")
            print("")
    
    if len(unique_files) < 2:
        print("No hay archivos únicos suficientes para la comparación de similitud.")
        return

    # --- FASE 2: Pre-procesar archivos únicos (Texto e Imagen) ---
    print(f"--- FASE 2: Pre-procesando {len(unique_files)} archivos únicos (extrayendo texto e imágenes) ---")
    print("Esto puede tardar varios minutos...")
    
    file_data = {} # {filepath: {"text": "...", "images": [np_array, ...]}}
    for filepath in unique_files:
        print(f"  Procesando: {os.path.basename(filepath)}")
        text, images = preprocess_pdf(filepath, render_dpi, compare_img_size, max_visual_pages)
        file_data[filepath] = {"text": text, "images": images}
        
    print("✅ Fase 2 completada.\n")

    # --- FASE 3: Comparar similitud (Texto y Visual) ---
    print("--- FASE 3: Comparando todos los pares (Similitud de Texto y Visual) ---")
    print("Esta es la fase más lenta...")

    text_similar_pairs = []
    visual_similar_pairs = []
    
    # Prepara el vectorizador de texto
    # Usamos stopwords en español definidas arriba para mejores resultados y evitar errores
    vectorizer = TfidfVectorizer(stop_words=SPANISH_STOPWORDS)

    # Itera sobre todas las combinaciones únicas de archivos
    for file1, file2 in combinations(unique_files, 2):
        data1 = file_data[file1]
        data2 = file_data[file2]

        # 1. Comparar Texto
        text_sim = compare_texts(data1["text"], data2["text"], vectorizer)
        if text_sim >= text_threshold:
            text_similar_pairs.append((file1, file2, text_sim))

        # 2. Comparar Imágenes (multi‑página)
        visual_sim = compare_images(data1["images"], data2["images"])
        if visual_sim >= visual_threshold:
            visual_similar_pairs.append((file1, file2, visual_sim))

    print("✅ Fase 3 completada.\n")

    # --- FASE 4: Reporte Final de Similitud ---
    print("--- REPORTE FINAL DE SIMILITUD ---")

    # Reporte de Similitud de Texto
    if not text_similar_pairs:
        print(f"✅ No se encontraron pares TEXTUALMENTE similares (Umbral > {text_threshold*100}%)")
    else:
        print(f"🚨 Se encontraron {len(text_similar_pairs)} pares TEXTUALMENTE similares:")
        text_similar_pairs.sort(key=lambda x: x[2], reverse=True)
        for f1, f2, sim in text_similar_pairs:
            print(f"\n  Similitud Texto: {sim:.2%}")
            print(f"    -> {f1}")
            print(f"    -> {f2}")

    print("\n" + "="*30 + "\n")

    # Reporte de Similitud Visual
    if not visual_similar_pairs:
        print(f"✅ No se encontraron pares VISUALMENTE similares (Umbral > {visual_threshold*100}%)")
    else:
        page_note = f"primeras {max_visual_pages} páginas" if max_visual_pages > 1 else "1ra página"
        print(f"🚨 Se encontraron {len(visual_similar_pairs)} pares VISUALMENTE similares ({page_note}):")
        visual_similar_pairs.sort(key=lambda x: x[2], reverse=True)
        for f1, f2, sim in visual_similar_pairs:
            print(f"\n  Similitud Visual: {sim:.2%}")
            print(f"    -> {f1}")
            print(f"    -> {f2}")

    # --- FASE 5: Exportar reportes ---
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)

            # Duplicados exactos
            dup_csv = os.path.join(out_dir, "duplicates.csv")
            with open(dup_csv, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["group_id", "file_path"])
                for i, group in enumerate(duplicate_groups, start=1):
                    for path in group:
                        w.writerow([i, path])

            # Similitud de texto
            text_csv = os.path.join(out_dir, "text_similar.csv")
            with open(text_csv, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["file1", "file2", "similarity"])
                for f1, f2, sim in sorted(text_similar_pairs, key=lambda x: x[2], reverse=True):
                    w.writerow([f1, f2, f"{sim:.6f}"])

            # Similitud visual
            visual_csv = os.path.join(out_dir, "visual_similar.csv")
            with open(visual_csv, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["file1", "file2", "similarity"])
                for f1, f2, sim in sorted(visual_similar_pairs, key=lambda x: x[2], reverse=True):
                    w.writerow([f1, f2, f"{sim:.6f}"])

            # Resumen JSON
            summary_json = os.path.join(out_dir, "summary.json")
            summary = {
                "directory": directory,
                "text_threshold": text_threshold,
                "visual_threshold": visual_threshold,
                "render_dpi": render_dpi,
                "compare_img_size": list(compare_img_size),
                "max_visual_pages": max_visual_pages,
                "duplicates": duplicate_groups,
                "text_similar": [
                    {"file1": f1, "file2": f2, "similarity": sim}
                    for f1, f2, sim in sorted(text_similar_pairs, key=lambda x: x[2], reverse=True)
                ],
                "visual_similar": [
                    {"file1": f1, "file2": f2, "similarity": sim}
                    for f1, f2, sim in sorted(visual_similar_pairs, key=lambda x: x[2], reverse=True)
                ],
            }
            with open(summary_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            # CSV consolidado
            consolidated_csv = os.path.join(out_dir, "report.csv")
            with open(consolidated_csv, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["type", "group_id", "file1", "file2", "similarity"])

                # duplicates (una fila por archivo en cada grupo)
                for i, group in enumerate(duplicate_groups, start=1):
                    for path in group:
                        w.writerow(["duplicate", i, path, "", ""])

                # text pairs
                for f1, f2, sim in sorted(text_similar_pairs, key=lambda x: x[2], reverse=True):
                    w.writerow(["text", "", f1, f2, f"{sim:.6f}"])

                # visual pairs
                for f1, f2, sim in sorted(visual_similar_pairs, key=lambda x: x[2], reverse=True):
                    w.writerow(["visual", "", f1, f2, f"{sim:.6f}"])

            print("\n📄 Reportes exportados en:")
            print(f"  - {dup_csv}")
            print(f"  - {text_csv}")
            print(f"  - {visual_csv}")
            print(f"  - {summary_json}")
            print(f"  - {consolidated_csv}")
        except Exception as e:
            print(f"[Error Reporte] No se pudieron exportar los reportes a '{out_dir}': {e}")

# --- Ejecutar el script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detecta PDFs duplicados o similares por texto e imagen.")
    parser.add_argument("--dir", dest="directory", default=TARGET_DIRECTORY, help="Carpeta objetivo con PDFs")
    parser.add_argument("--text-threshold", type=float, default=TEXT_SIMILARITY_THRESHOLD, help="Umbral similitud de texto (0-1)")
    parser.add_argument("--visual-threshold", type=float, default=VISUAL_SIMILARITY_THRESHOLD, help="Umbral similitud visual (0-1)")
    parser.add_argument("--dpi", dest="dpi", type=int, default=RENDER_DPI, help="DPI para renderizar páginas")
    parser.add_argument("--img-size", dest="img_size", nargs=2, type=int, default=list(COMPARE_IMG_SIZE), metavar=("W","H"), help="Tamaño estándar W H para comparar imágenes")
    parser.add_argument("--visual-pages", dest="visual_pages", type=int, default=MAX_VISUAL_PAGES, help="Número de páginas para similitud visual")
    parser.add_argument("--out-dir", dest="out_dir", default=os.path.join(os.getcwd(), "reports"), help="Directorio de salida para reportes CSV/JSON")
    parser.add_argument("--include-dirs", dest="include_dirs", default="", help="Lista separada por comas de subcadenas a incluir en rutas de directorio")
    parser.add_argument("--exclude-dirs", dest="exclude_dirs", default="", help="Lista separada por comas de subcadenas a excluir en rutas de directorio")

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: La ruta '{args.directory}' no existe o no es un directorio.")
    else:
        # Parsear listas de filtros
        include_dirs = [s.strip() for s in args.include_dirs.split(',') if s.strip()] if args.include_dirs else []
        exclude_dirs = [s.strip() for s in args.exclude_dirs.split(',') if s.strip()] if args.exclude_dirs else []
        find_all_similar_pdfs(
            directory=args.directory,
            text_threshold=args.text_threshold,
            visual_threshold=args.visual_threshold,
            render_dpi=args.dpi,
            compare_img_size=(args.img_size[0], args.img_size[1]),
            max_visual_pages=args.visual_pages,
            out_dir=args.out_dir,
            include_dirs=include_dirs,
            exclude_dirs=exclude_dirs,
        )