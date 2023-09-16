# Scientific Papers Trend Analysis

This project aims to provide insights into current trends in science based on a large archive of scientific papers. The analysis involves data preprocessing, dimensionality reduction, clustering, and visualization.

## Repository Structure

```
project/
│
├── data/
│   ├── arxiv-metadata-oai.json        # Original dataset
│   └── data_preprocessed.csv          # Cleaned and processed data
│   └── data_clustered.csv             # Clustered data

├── images/                            # Generated visualizations
│
├── notebooks/
│   └── insights.ipynb                 # Jupyter notebook for EDA and initial insights
│   └── preprocessing.ipynb            # Jupyter notebook for data preprocessing
│   └── clustering.ipynb               # Jupyter notebook for clustering
│   └── clustering.ipynb               # Jupyter notebook for analysis
│
│
├── requirements.txt                   # Required libraries
└── README.md                          # Project description and instructions
```

## Instructions to Run the Code

1. Clone the repository to your local machine.

2. Install the required libraries by running the following command in the terminal:
```
pip install -r requirements.txt
```

3. Run the notebooks in the following order:
```
insights.ipynb
preprocessing.ipynb
clustering.ipynb
analysis.ipynb
```

## Author

[Mohammadsadegh Solouki]

## Acknowledgments

This project uses the [arXiv](https://arxiv.org/) dataset.

---
