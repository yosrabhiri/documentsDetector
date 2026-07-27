# SplitDocument

Application de traitement de PDF contenant plusieurs documents mélangés.

## Module 1 : analyse avant OCR

Ce premier module vérifie le PDF, compte ses pages et détermine quelles pages
nécessitent un OCR. Une page est considérée comme scannée lorsqu'elle contient
moins de 20 caractères textuels utiles.

### Installation

```powershell
python -m pip install -r requirements.txt
```

Pour contribuer au projet et lancer les tests :

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

### Exécution

```powershell
python main.py analyze "samples\EXPERT TRAVEL SERVICES ( DOC JUR ).pdf"
```

Le rapport est créé dans `output/<reference>/analysis.json`. Il contient les
dimensions, le nombre d'images, la quantité de texte et la décision OCR pour
chaque page.

## Module 2 : OCR francais-arabe

Tesseract traite localement les pages sans texte avec les modeles `fra+ara`.

```powershell
python main.py ocr "samples\EXPERT TRAVEL SERVICES ( DOC JUR ).pdf"
```

Pour tester seulement quelques pages :

```powershell
python main.py ocr "samples\EXPERT TRAVEL SERVICES ( DOC JUR ).pdf" --pages 1,32,36
```

Les textes et le rapport sont enregistres dans `output/<reference>/ocr/`.

## Module 3 : classification des pages

```powershell
python main.py classify "output\EXPERT TRAVEL SERVICES ( DOC JUR )\ocr"
```

La classification est independante de l'ordre des documents et fournit les
mots-cles trouves ainsi qu'un score de confiance.

### Detection recto-verso d'une CIN

```powershell
python main.py detect-cin "samples\document.pdf" --page 12
```

Le rapport indique uniquement les faces, les indices techniques et la
confiance. Il ne conserve ni texte OCR brut, ni nom, ni numero d'identite.

Pour rechercher automatiquement les CIN parmi toutes les pages inconnues :

```powershell
python main.py auto-detect-cin "samples\document.pdf" "output\REFERENCE\ocr"
```

## Module 4 : segmentation et creation des PDF

```powershell
python main.py split "output\EXPERT TRAVEL SERVICES ( DOC JUR )\ocr"
```

Les documents sont crees dans `output/<reference>/documents/`. Les segments
inconnus ne sont pas exportes; les faibles confiances sont exportees et signalees.

### Second OCR cible

```powershell
python main.py refine "output\EXPERT TRAVEL SERVICES ( DOC JUR )\ocr"
```

Seules les classifications connues sous le seuil de confiance sont retraitees.
Le texte est remplace uniquement si le type reste identique et le score augmente.
