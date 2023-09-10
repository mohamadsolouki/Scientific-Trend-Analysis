# Scientific Papers Trend Analysis

This project aims to provide insights into current trends in science based on a large archive of scientific papers. The analysis involves data preprocessing, dimensionality reduction, clustering, and visualization.

## Repository Structure

```
project/
│
├── data/
│   ├── arxiv-metadata-oai.json  # Original dataset
│   └── data_preprocessed.csv                    # Cleaned and processed data
│   └── data_clustered.csv                    # Clustered data
│   └── data_concatenated.csv                    # Concatenated data
│
├── notebooks/
│   └── analysis.ipynb               # Jupyter notebook for analysis
│   └── clustering.ipynb               # Jupyter notebook for analysis
│
├── scripts/
│   └── preprocess.py                # Script for data preprocessing
│   └── clustering.py                # Script for data clustering
│
├── requirements.txt                      # Required libraries
└── README.md                             # Project description and instructions
```

## Instructions to Run the Code

1. Clone the repository to your local machine.

2. Install the required libraries by running the following command in the terminal:
```
pip install -r requirements.txt
```

3. Run the `preprocess_data.py` script to preprocess the data, perform dimensionality reduction, and apply clustering:
```
python scripts/preprocess_data.py
```

4. Open the `main_notebook.ipynb` notebook and run the cells to perform further analysis and generate visualizations.

5. Interpret the results and provide insights into the current trends in science based on the clusters.

## Author

[Mohammadsadegh Solouki]

## Acknowledgments

This project was uses the [arXiv](https://arxiv.org/) dataset.

---