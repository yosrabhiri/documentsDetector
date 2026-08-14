# SplitDocument

Application locale qui analyse des PDF numérisés contenant plusieurs documents,
reconnaît leur type et crée un fichier PDF séparé pour chaque document détecté.
Elle prend en charge un PDF unique ou un lot de PDF, avec OCR français-arabe.

## Prérequis

Vérifier les installations dans PowerShell :

```powershell
python --version
git --version
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --version
```

## Installation

Cloner le dépôt et entrer dans le projet :

```powershell
git clone https://github.com/yosrabhiri/documentsDetector.git
cd documentsDetector
```

Créer puis activer un environnement Python isolé :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Installer les dépendances :

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell refuse l'activation de l'environnement, exécuter une fois dans la
même fenêtre :

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Lancer l'interface

Depuis la racine du projet :

```powershell
python -m streamlit run app.py
```

Ouvrir ensuite `http://127.0.0.1:8501` dans le navigateur. Le terminal doit
rester ouvert pendant l'utilisation. Pour arrêter le serveur, utiliser `Ctrl+C`.

L'interface propose deux workflows :

- `Un PDF` : traiter et télécharger les documents extraits d'un seul PDF.
- `Traitement par lot` : sélectionner plusieurs PDF ou un dossier complet.

Dans un lot, le nom de chaque PDF devient sa référence. Deux fichiers portant le
même nom ne peuvent donc pas être traités dans le même lot.

## Résultats

Chaque référence reçoit son propre dossier :

```text
output/
└── REFERENCE/
    ├── ocr/
    │   ├── page_0001.txt
    │   ├── ocr.json
    │   └── classification.json
    ├── documents/
    │   ├── contrat_REFERENCE.pdf
    │   ├── cin_REFERENCE.pdf
    │   └── statuts_REFERENCE.pdf
    ├── analysis.json
    ├── cin_detection.json
    ├── segmentation.json
    └── traitement.json
```

- `ocr/` contient le texte extrait et la classification de chaque page.
- `documents/` contient les PDF séparés finaux.
- `analysis.json` décrit les pages et indique celles qui nécessitent un OCR.
- `cin_detection.json` contient les résultats de détection des CIN.
- `segmentation.json` décrit les groupes de pages détectés.
- `traitement.json` est le rapport final avec les durées et alertes de révision.

Les entrées confidentielles, résultats et fichiers temporaires sont ignorés par
Git et ne sont pas publiés dans le dépôt.

## Ligne de commande

Traiter un PDF de bout en bout :

```powershell
python main.py process "C:\chemin\REFERENCE.pdf"
```

Traiter tous les PDF d'un dossier :

```powershell
python main.py process-batch --folder "C:\chemin\dossier"
```

Traiter une liste précise :

```powershell
python main.py process-batch "C:\docs\REF001.pdf" "C:\docs\REF002.pdf"
```

Quatre pages sont traitées simultanément par défaut. Pour réduire la charge :

```powershell
python main.py process "C:\chemin\REFERENCE.pdf" --ocr-workers 2
```

## Commandes par module

Analyser un PDF avant OCR :

```powershell
python main.py analyze "C:\chemin\REFERENCE.pdf"
```

Extraire le texte français-arabe de toutes les pages ou d'une sélection :

```powershell
python main.py ocr "C:\chemin\REFERENCE.pdf"
python main.py ocr "C:\chemin\REFERENCE.pdf" --pages 1,3-5
```

Classifier les textes OCR, améliorer les pages peu fiables et créer les PDF :

```powershell
python main.py classify "output\REFERENCE\ocr"
python main.py refine "output\REFERENCE\ocr"
python main.py split "output\REFERENCE\ocr"
```

Détecter une CIN sur une page précise ou automatiquement parmi les pages
inconnues :

```powershell
python main.py detect-cin "C:\chemin\REFERENCE.pdf" --page 12
python main.py auto-detect-cin "C:\chemin\REFERENCE.pdf" "output\REFERENCE\ocr"
```

## Validation et tests

Installer les dépendances de développement puis lancer les tests :

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Créer des variantes locales d'un PDF déjà segmenté :

```powershell
python main.py generate-scenarios "C:\chemin\REFERENCE.pdf" `
  "output\REFERENCE\segmentation.json"
```

Comparer une segmentation obtenue avec le résultat attendu :

```powershell
python main.py evaluate attendu.json segmentation.json
```

Les scénarios sont créés dans `validation_runs/`, qui est ignoré par Git.

## Dépannage

- `Tesseract executable not found` : installer Tesseract dans le chemin indiqué
  dans les prérequis.
- Port `8501` occupé : lancer
  `python -m streamlit run app.py --server.port 8502` puis ouvrir
  `http://127.0.0.1:8502`.
- Deux fichiers ont le même nom : supprimer le doublon ou renommer l'un des PDF
  avant de relancer le lot.
- Pour les PDF volumineux, vérifier que le fichier ne dépasse pas la limite
  d'import de 500 Mo configurée dans `.streamlit/config.toml`.
