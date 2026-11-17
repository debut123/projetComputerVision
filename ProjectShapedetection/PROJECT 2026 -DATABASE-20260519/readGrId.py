import numpy as np
import scipy.optimize as sp
import skimage.io as ski 
import os
from skimage import color
from skimage import filters
from skimage import feature
from skimage.transform import hough_line, hough_line_peaks, rotate
from skimage import feature, morphology
from scipy.ndimage import binary_fill_holes
import skimage
from scipy import ndimage

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
    # ✅ 4. MATRICE
    # --------------------------
    matrice = np.zeros((len(rows), len(rows[0])))

    scores_list = []

    for i, row in enumerate(rows):
        for j, b in enumerate(row):
            
            k, size, cx, cy, min_row, min_col, max_row, max_col = b
            
            case = result[min_row:max_row, min_col:max_col]

            # ✅ Sobel marche mieux chez toi
            edges = filters.sobel(case)
            score = np.sum(edges)

            scores_list.append((score, i, j))

    # --------------------------
    # ✅ 5. SEUIL ADAPTÉ
    # --------------------------
    scores_only = [s for s, _, _ in scores_list]
    threshold_score = np.mean(scores_only) + 0.5*np.std(scores_only)

    selected = [(s, i, j) for (s, i, j) in scores_list if s > threshold_score]

    selected.sort(reverse=True, key=lambda x: x[0])
    selected = selected[:8]

    # --------------------------
    # ✅ 6. REMPLISSAGE MATRICE
    # --------------------------
    for score, i, j in selected:
        if i < matrice.shape[0] and j < matrice.shape[1]:
            matrice[i, j] = 1

    # --------------------------
    # ✅ 7. LECTURE
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






    