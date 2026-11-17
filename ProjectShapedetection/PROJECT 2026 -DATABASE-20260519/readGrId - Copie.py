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
    im_rgb = im[:, :, :3]
    im2 = color.rgb2gray(im_rgb)
    im3 = filters.sobel(im2)
    size = 5
    seuil = filters.threshold_otsu(im3)
    im4 = im3>seuil*0.7
    im35=im4.copy()
    im4 = morphology.remove_small_objects(im4, min_size=100)
    im_dilate = ndimage.binary_dilation(im4, structure=np.ones((size, size)))
    im_close = morphology.binary_closing(im_dilate, morphology.square(size))
    im4 = binary_fill_holes(im_close)
    imlabel, nbCC   = skimage.measure.label(im4, connectivity=2, return_num= True)
    regionprops= skimage.measure.regionprops(imlabel)
    dimMax = []
    for k in range(len(regionprops)):

        min_row, min_col, max_row, max_col = regionprops[k].bbox
        
        pattern = imlabel[min_row:max_row, min_col:max_col] > 0
        
        dimL, diml = pattern.shape
        size = dimL * diml
        nbones = np.sum(pattern)
        density = nbones / size
        
        ratio = dimL / diml

        if density > 0.9 and 0.8 < ratio < 1.25:
            dimMax.append([k, size])
    if len(dimMax)<80:
        return False , False
    # --- 2. Filtre taille (relatif) ---
    case = np.array(dimMax)
    median_size = np.median(case[:, 1])

    dimMax = [x for x in dimMax if abs(x[1] - median_size) < 0.15 * median_size]

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
    d_typical = np.median(distances)

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
    threshold = filters.threshold_otsu(im2)

    # binarisation + découpage
    result = (im2 > threshold)

    # construire liste avec coordonnées
    boxes = []

    for k, size in filtered:
        min_row, min_col, max_row, max_col = regionprops[k].bbox
        
        cx = (min_col + max_col) / 2
        cy = (min_row + max_row) / 2

        boxes.append((k, size, cx, cy, min_row, min_col, max_row, max_col))

    # tri par ligne puis colonne
    boxes = sorted(boxes, key=lambda x: (x[3], x[2]))  # (cy, cx)
    # --- regrouper les lignes ---
    rows = []
    tolerance = 20  # ajuste si besoin

    for b in boxes:
        placed = False
        for row in rows:
            # comparer avec le premier élément de la ligne
            if abs(b[3] - row[0][3]) < tolerance:
                row.append(b)
                placed = True
                break
        
        if not placed:
            rows.append([b])

    # trier chaque ligne par colonne
    for row in rows:
        row.sort(key=lambda x: x[2])  # tri par cx

    # trier les lignes entre elles
    # rows.sort(key=lambda r: r[0][3])
    rows.sort(key=lambda r: np.mean([b[3] for b in r]))
    nbbare= 0 
    matrice = np.zeros((len(rows), len(rows[0])))  # matrice lignes x colonnes
    for i, row in enumerate(rows):
        for j, b in enumerate(row):
            
            k, size, cx, cy, min_row, min_col, max_row, max_col = b
            
            case = result[min_row:max_row, min_col:max_col]
            
            # gradient pour voir les traits
            edges = filters.sobel(case)

            # mesure d'activité (présence de traits)
            score = np.sum(edges)

            # seuil (à ajuster)
            if score > 700:
                
                nbbare += 1
                matrice[i, j] = 1
    mat_groupe = matrice[:, :3]
    mat_id = matrice[:, 3:]

    alphabet = "ABCDEFGHIJ"

    # --- groupe ---
    groupe = ""
    for j in range(mat_groupe.shape[1]):
        col = mat_groupe[:, j]
        idx = np.argmax(col)

        if j < 2:
            groupe += str(idx)
        else:
            groupe += alphabet[idx]
    # --- ID ---
    identifiant = ""
    for j in range(mat_id.shape[1]):
        col = mat_id[:, j]
        idx = np.argmax(col)
        identifiant += str(idx)
    return groupe, identifiant