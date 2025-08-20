## Project Overview
This project is based on the [ODIR-5K](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k) fundus image dataset, focusing on:
1. **Multi-class fundus disease classification** (8 classes)
2. **Conditional generative models** (cGAN / DDIM + CFG)

Objective: Evaluate classification performance using Accuracy, Precision, Recall, F1-score, and AUC, and assess generated image quality using FID, KID, IS, and semantic consistency.

---
## Data
- The dataset contains color retinal images and corresponding labels, collected from multiple centers.
- Data processing:
  - De-identification: only patient ID, eye side, and disease label are retained
  - 5-fold split: train/validation/test = 60/20/20%
  - Test set is used only once for final evaluation
- CSV file: `./data/csv/Overall_label.csv`

---
## File Structure
After downloading the dataset, organize it in the following structure
```text
ODIR_project/
│
├── data/
│ │ 
│ ├── ODIR-5K/Training Images/
│ ├── dataset_utils.py
│ ├── pro_csv.py
│ └── pro_csv.sh
├── notebook/
│ ├──figure/ 
│ └── EDA.ipynb
├── experiments/
│ └──configs/ 
│   ├──Classfication.yaml
│   ├──par.py
│   └──Generation.yaml
├── src/
│ ├── model/
│ ├── Classification_training.py
│ ├── Generation_evaluation.py
│ ├── Generation_evaluation.sh
│ ├── Generation_training.py
│ └── utils.py
├── requirements.txt
└── README.md
```

---
## Installation & Dependencies
```bash
conda create -n odir python=3.12 -y
conda activate odir
pip install -r requirements.txt
```
---
## Usage
1. Data Processing
```bash
sh data/pro_csv.sh 
```

2. Data Exploration and Visualization
```bash
jupyter notebook notebook/EDA.ipynb
```

3. Classification Training

```bash
#Modify parameters in experiments/configs/Classification.yaml.
python src/Classification_training.py
```

4. Conditional Generation training

```bash
#Modify parameters in experiments/configs/Generation.yaml
python src/Generation_training.py
```

5. Evaluate Generative model

```bash
#Modify parameters in src/Generation_evaluation.sh. Evaluation of the generative models requires manually specifying parameters.
bash Generation_evaluation.sh
```

---

## Notes

1. The data files are large; please download the ODIR-5K dataset yourself and place it in the data/ directory.

2. Evaluation of the generative models requires manually specifying parameters.