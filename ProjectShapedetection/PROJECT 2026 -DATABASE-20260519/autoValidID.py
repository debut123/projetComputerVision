# !pip install numpy
# !pip install pandas
# !pip install skimage
# !pip install sklearn

import os
import numpy as np
import pandas as pd
import skimage.io as ski
from skimage import color, filters, transform, morphology, measure
from sklearn.metrics.pairwise import cosine_similarity
from skimage import feature, morphology
from scipy.ndimage import binary_fill_holes
import skimage
from skimage.transform import hough_line, hough_line_peaks, rotate
# --------------------------
#  PREPROCESS SIGNATURE
# --------------------------
def preprocess_signature(img):

    if len(img.shape) == 3:
        img = color.rgb2gray(img)

    img = (img - np.mean(img)) / (np.std(img) + 1e-8)

    th = filters.threshold_otsu(img)
    binary = img < th

    binary = morphology.binary_dilation(binary, morphology.disk(2))
    binary = morphology.remove_small_objects(binary, min_size=200)

    labels = morphology.label(binary)
    regions = measure.regionprops(labels)

    if len(regions) == 0:
        return np.zeros((64*128,))

    largest = max(regions, key=lambda r: r.area)

    minr, minc, maxr, maxc = largest.bbox
    crop = binary[minr:maxr, minc:maxc]

    out = np.zeros((64, 128))

    h, w = crop.shape
    scale = min(64/h, 128/w)

    new_h = int(h * scale)
    new_w = int(w * scale)

    resized = transform.resize(crop, (new_h, new_w))

    y0 = (64 - new_h)//2
    x0 = (128 - new_w)//2

    out[y0:y0+new_h, x0:x0+new_w] = resized

    feat = out.flatten()
    feat = feat / (np.linalg.norm(feat) + 1e-8)

    return feat


# --------------------------
#  LOAD DATABASE
# --------------------------
def load_signature_db(base_path):

    database = {}

    for student_id in os.listdir(base_path):
        person_path = os.path.join(base_path, student_id)

        if not os.path.isdir(person_path):
            continue

        feats = []

        for file in os.listdir(person_path):
            if file.endswith(".png"):
                img = ski.imread(os.path.join(person_path, file))
                feat = preprocess_signature(img)

                if np.mean(feat) < 1e-3:
                    continue

                feats.append(feat)

        if len(feats) > 0:
            database[student_id] = np.array(feats)

    return database


# --------------------------
#  MATCH SIGNATURE → ID
# --------------------------
def match_signature(signature_img, database):

    test_feat = preprocess_signature(signature_img)

    best_id = None
    best_score = -1

    for student_id, X in database.items():

        sims = cosine_similarity([test_feat], X)[0]

        top_k = min(5, len(sims))
        best_scores = np.sort(sims)[-top_k:]

        score = 0.7*np.mean(best_scores) + 0.3*np.max(best_scores)

        mean_X = np.mean(X, axis=0)
        dist = np.linalg.norm(test_feat - mean_X)

        score = score - 0.02 * dist

        if score > best_score:
            best_score = score
            best_id = student_id

    return best_id, best_score


# --------------------------
#  EXTRACTION SIGNATURE (TA VERSION SIMPLIFIÉE)
# --------------------------
def extract_signature(im):

    if im.shape[1] > im.shape[0]:
        im = transform.rotate(im, -90)

    im = color.rgb2gray(im[:, :, :3])

    im3 = filters.sobel(im)

    seuil = filters.threshold_otsu(im3)
    im4 = im3 > seuil * 0.7

    im4 = morphology.remove_small_objects(im4, min_size=100)
    im4 = morphology.binary_closing(im4)
    im4 = morphology.remove_small_holes(im4)

    imlabel = measure.label(im4)
    regions = measure.regionprops(imlabel)

    best = None
    best_area = 0

    for r in regions:
        minr, minc, maxr, maxc = r.bbox
        area = (maxr-minr)*(maxc-minc)

        if area > best_area and area < 0.2*im.shape[0]*im.shape[1]:
            best_area = area
            best = r.bbox

    if best is None:
        return None

    minr, minc, maxr, maxc = best
    signature = im[minr:maxr, minc:maxc]

    p = 20
    if signature.shape[0] > 2*p and signature.shape[1] > 2*p:
        signature = signature[p:-p, p:-p]

    return signature


# --------------------------
#  FONCTION PRINCIPALE DEMANDÉE
# --------------------------
def autoValidID(filename, SIGN_DB_PATH, EXCEL_PATH, readGroupeID_func):

    im = ski.imread(filename)

    #  1. ID grille
    groupe, student_grid = readGroupeID_func(im)

    if student_grid is False:
        print("❌ lecture grille échouée")
        return

    #  2. signature extraite
    signature = extract_signature(im)

    if signature is None:
        print("❌ signature non détectée")
        return

    #  3. database
    database = load_signature_db(SIGN_DB_PATH)

    #  4. matching
    student_bitmap, score = match_signature(signature, database)

    print("\nRésultat :")
    print("ID grille   :", student_grid)
    print("ID signature:", student_bitmap)
    print("score       :", score)

    #  5. Excel
    if os.path.exists(EXCEL_PATH):
        df = pd.read_excel(EXCEL_PATH)
    else:
        df = pd.DataFrame(columns=["StudentID_grid", "StudentID_bitmap", "Valid"])

    valid = (student_grid == student_bitmap)

    new_row = {
        "StudentID_grid": student_grid,
        "StudentID_bitmap": student_bitmap,
        "Valid": valid
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_excel(EXCEL_PATH, index=False)

    print(" ligne ajoutée Excel")


def readGroupeID(im):
    if im.shape[1]>im.shape[0]:
        im = rotate(im, -90)
    im_rgb = im[:, :, :3]  # on garde uniquement R, G, B
    im2 = color.rgb2gray(im_rgb)

    # dir(skimage.filters)
    im3 = filters.sobel(im2)


    size = 2
    seuil = filters.threshold_otsu(im3)
    im4 = im3>seuil*0.7
    # im35=im4.copy()
    im4 = morphology.remove_small_objects(im4, min_size=100)
    im4 = binary_fill_holes(im4)
    imlabel, nbCC   = skimage.measure.label(im4, connectivity=2, return_num= True)
    regionprops= skimage.measure.regionprops(imlabel)
        # --- 1. Filtrage initial (densité + forme) ---

    dimMax = []

    for k in range(len(regionprops)):

        min_row, min_col, max_row, max_col = regionprops[k].bbox
    
        pattern = imlabel[min_row:max_row, min_col:max_col] > 0
        
        dimL, diml = pattern.shape
        size = dimL * diml
        nbones = np.sum(pattern)
        density = nbones / size
        
        ratio = dimL / diml
        
        perimeter = regionprops[k].perimeter
        compactness = size / (perimeter ** 2)


        if density > 0.6 and 0.6 < ratio < 1.5 and compactness > 0.05:
            dimMax.append([k, size])
    if len(dimMax)<80:
        return False , False
    # --- 2. Filtre taille (relatif) ---
    case = np.array(dimMax)
    median_size = np.median(case[:, 1])

    min_size = 0.6 * median_size   # rejette les petits
    max_size = 1.7 * median_size   # accepte plus grand

    dimMax = [x for x in dimMax if min_size < x[1] < max_size]

    # --- 3. Centres ---
    centers = []
    for k, size in dimMax:
        min_row, min_col, max_row, max_col = regionprops[k].bbox
        cx = (min_col + max_col) / 2
        cy = (min_row + max_row) / 2
        centers.append((k, cx, cy, size))

    centers_xy = np.array([(cx, cy) for (_, cx, cy, _) in centers])
    if len(centers_xy)<80:
        return False , False

    # --- 4. Distance typique ---
    D = np.sqrt(
        (centers_xy[:, None, 0] - centers_xy[None, :, 0])**2 +
        (centers_xy[:, None, 1] - centers_xy[None, :, 1])**2
    )

    distances = D[D > 0]
    d_typical = np.min([np.median(distances), 250])

    # --- 5. Filtrage par cohérence de grille ---
    filtered = []

    for i in range(len(centers)):
        k, cx, cy, size = centers[i]
        
        dists = D[i]
        
        neighbors = np.sum(
            (dists > 0.7 * d_typical) &
            (dists < 1.3 * d_typical)
        )
        
        if neighbors >= 3:
            filtered.append((k, size))
            # seuillage Otsu
    # --------------------------
    #  1. BINARISATION
    # --------------------------
    threshold = filters.threshold_otsu(im2)
    result = (im2 > threshold)

    # --------------------------
    #  2. EXTRACTION BOXES
    # --------------------------
    boxes = []

    for k, size in filtered:
        min_row, min_col, max_row, max_col = regionprops[k].bbox
        
        cx = (min_col + max_col) / 2
        cy = (min_row + max_row) / 2

        boxes.append((k, size, cx, cy,
                    int(min_row), int(min_col),
                    int(max_row), int(max_col)))

    # --------------------------
    #  3. TRI + REGROUPEMENT LIGNES
    # --------------------------
    boxes = sorted(boxes, key=lambda x: (x[3], x[2]))



    rows = []

    for i in range(0, len(boxes), 8):
        row = boxes[i:i+8]
        row.sort(key=lambda x: x[2])  # sécuriser ordre horizontal
        rows.append(row)


        
    if len(rows) != 10 or any(len(row) != 8 for row in rows):
        return False , False

    # tri interne
    for row in rows:
        row.sort(key=lambda x: x[2])

    rows.sort(key=lambda r: np.mean([b[3] for b in r]))

    # --------------------------
    #  4. MATRICE
    # --------------------------
    matrice = np.zeros((len(rows), len(rows[0])))

    scores_list = []

    for i, row in enumerate(rows):
        for j, b in enumerate(row):
            
            k, size, cx, cy, min_row, min_col, max_row, max_col = b
            
            case = result[min_row:max_row, min_col:max_col]

            #  Sobel marche mieux chez toi
            edges = filters.sobel(case)
            score = np.sum(edges)

            scores_list.append((score, i, j))

    # --------------------------
    #  5. SEUIL ADAPTÉ
    # --------------------------
    scores_only = [s for s, _, _ in scores_list]
    threshold_score = np.mean(scores_only) + 0.5*np.std(scores_only)

    selected = [(s, i, j) for (s, i, j) in scores_list if s > threshold_score]

    selected.sort(reverse=True, key=lambda x: x[0])
    selected = selected[:8]

    # --------------------------
    #  6. REMPLISSAGE MATRICE
    # --------------------------
    for score, i, j in selected:
        if i < matrice.shape[0] and j < matrice.shape[1]:
            matrice[i, j] = 1

    # --------------------------
    #  7. LECTURE
    # --------------------------
    mat_groupe = matrice[:, :3]
    mat_id = matrice[:, 3:]

    alphabet = "ABCDEFGHIJ"

    groupe = ""
    for j in range(mat_groupe.shape[1]):
        idx = np.argmax(mat_groupe[:, j])
        groupe += str(idx) if j < 2 else alphabet[idx]

    identifiant = ""
    for j in range(mat_id.shape[1]):
        idx = np.argmax(mat_id[:, j])
        identifiant += str(idx)

    return groupe, identifiant



# --------------------------
#  FONCTION PRINCIPALE
# --------------------------
def autoValidPresences(EXAM_FORMXX_PRESENCES,
                       STUDENT_CLASS_SIGNATURES,
                       EXAM_FORMXX_RESULTS,
                       readGroupeID_func,
                       extract_signature_func,
                       match_signature_func,
                       load_signature_db_func):

    #  chemin fichier Excel résultat
    excel_path = os.path.join(EXAM_FORMXX_RESULTS, "EXAM_FORMXX_PRESENCES.xlsx")

    #  créer dataframe vide
    df = pd.DataFrame(columns=["imageName", "studentID_grid", "studentID_signature"])

    #  charger base signatures UNE SEULE FOIS
    database = load_signature_db_func(STUDENT_CLASS_SIGNATURES)

    # --------------------------
    #  PARCOURS DES IMAGES
    # --------------------------
    for filename in os.listdir(EXAM_FORMXX_PRESENCES):

        if not (filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg")):
            continue

        path = os.path.join(EXAM_FORMXX_PRESENCES, filename)

        print(f"\n📄 Traitement : {filename}")

        try:
            im = ski.imread(path)

            #  1. ID grille
            groupe, student_grid = readGroupeID_func(im)

            if student_grid is False:
                print("❌ lecture grille échouée")
                student_grid = ""

            #  2. extraction signature
            signature = extract_signature_func(im)

            if signature is None:
                print("❌ signature non détectée")
                student_signature = ""
            else:
                #  3. matching signature
                student_signature, score = match_signature_func(signature, database)

                print("→ ID signature :", student_signature)
                print("→ score :", score)

                if score < 0.22:
                    student_signature = ""  # non reconnu

            #  4. ajout ligne Excel
            new_row = {
                "imageName": filename,
                "studentID_grid": student_grid,
                "studentID_signature": student_signature
            }

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        except Exception as e:
            print("❌ erreur :", e)

    # --------------------------
    #  SAUVEGARDE
    # --------------------------
    #os.makedirs(EXAM_FORMXX_RESULTS, exist_ok=True)
    
    # if not os.path.exists(EXAM_FORMXX_RESULTS):
    #     os.makedirs(EXAM_FORMXX_RESULTS)

    # df.to_excel(excel_path, index=False)

    # print("\n✅ Fichier généré :", excel_path)

    
    # # ✅ créer dossier si besoin
    # os.makedirs(EXAM_FORMXX_RESULTS, exist_ok=True)
    
    # # ✅ extraire numéro du formulaire (ex: FORM1 → 01)
    # form_name = os.path.basename(EXAM_FORMXX_PRESENCES)
    # form_num = ''.join([c for c in form_name if c.isdigit()]).zfill(2)
    
    # # ✅ nom du fichier correct
    # excel_filename = f"EXAM_FORM{form_num}_PRESENCES.xlsx"
    
    # excel_path = os.path.join(EXAM_FORMXX_RESULTS, excel_filename)
    
    # # ✅ sauvegarde
    # df.to_excel(excel_path, index=False)
    EXAM_FORMXX_RESULTS = r"C:\Users\frdch\Documents\Francois\Classe\ISEP\cycleIngénieur\2meAnnee\Vision par ordinateur\PROJECT 2026 - EXAM FORM RECOGNITION DATABASE-20260330\ProjectShapedetection\PROJECT 2026 -DATABASE-20260519\EXAM_FORM01_RESULTS"
    # os.makedirs(EXAM_FORMXX_RESULTS, exist_ok=True)
    
    # excel_path = os.path.join(EXAM_FORMXX_RESULTS, "EXAM_FORM01_PRESENCES.xlsx")
    
    # df.to_excel(excel_path, index=False)
    
       
    os.makedirs(EXAM_FORMXX_RESULTS, exist_ok=True)
    
    excel_path = os.path.join(EXAM_FORMXX_RESULTS, "EXAM_FORM01_PRESENCES.xlsx")
    
    # supprime ancien fichier
    if os.path.exists(excel_path):
        os.remove(excel_path)
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    
    print("✅ Excel créé :", excel_path)
    



# print("✅ Fichier généré :", excel_path)

autoValidPresences(
    EXAM_FORMXX_PRESENCES=r"C:\Users\frdch\Documents\Francois\Classe\ISEP\cycleIngénieur\2meAnnee\Vision par ordinateur\PROJECT 2026 - EXAM FORM RECOGNITION DATABASE-20260330\ProjectShapedetection\PROJECT 2026 -DATABASE-20260519\FORM1",
    STUDENT_CLASS_SIGNATURES=r"C:\Users\frdch\Documents\Francois\Classe\ISEP\cycleIngénieur\2meAnnee\Vision par ordinateur\PROJECT 2026 - EXAM FORM RECOGNITION DATABASE-20260330\ProjectShapedetection\PROJECT 2026 -DATABASE-20260519\SIGNATURES\Scan_16042026093714_SIG",
    EXAM_FORMXX_RESULTS=r"C:\Users\frdch\Documents\Francois\Classe\ISEP\cycleIngénieur\2meAnnee\Vision par ordinateur\PROJECT 2026 - EXAM FORM RECOGNITION DATABASE-20260330\ProjectShapedetection\PROJECT 2026 -DATABASE-20260519\EXAM_FORM01_RESULTS",
    readGroupeID_func=readGroupeID,
    extract_signature_func=extract_signature,
    match_signature_func=match_signature,
    load_signature_db_func=load_signature_db
)
