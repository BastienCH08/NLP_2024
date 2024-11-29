# NLP for TripAdvisor Recommendation Challenge

This repository contains a Jupyter Notebook focused on solving the TripAdvisor Recommendation Challenge using Natural Language Processing (NLP) techniques. It integrates various stages of preprocessing, data handling, and model implementation to generate recommendations based on user reviews.

The objective of this challenge is to create a unique recommendation system for TripAdvisor reviews. This involves utilizing a range of pre-treatments and vocabulary manipulation techniques on the reviews. Both pre-trained machine learning models and custom-developed models will be employed as needed. By integrating various methodologies, this approach aims to optimize the recommendation process while adhering to the constraint of not using direct supervised learning to predict the best place based on a specific query.

## Project Overview
The objective of this notebook is to apply NLP techniques to analyze user reviews and recommend hotels. Specifically, it explores the BM25 algorithm and compares it with a custom-built model.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Notebook Features](#notebook-features)
3. [Dependencies](#dependencies)
4. [Data Preprocessing](#data-preprocessing)
5. [Modeling](#modeling)
6. [Results](#results)
7. [How to Use](#how-to-use)
8. [Acknowledgments](#acknowledgments)
9. [Authors](#authors)

## Getting Started
To get started, clone this repository and open the notebook in your preferred Jupyter environment. Make sure all dependencies are installed (see [Dependencies](#dependencies)).

## Notebook Features
- **Natural Language Processing (NLP):** Implements core NLP tasks to preprocess and analyze text data from TripAdvisor.
- **BM25 Algorithm:** Uses the `rank_bm25` library to perform relevance scoring for recommendations.
- **Custom Model Comparison:** Benchmarks BM25 against a custom model developed in the notebook.
- **Data Handling:** Downloads and preprocesses hotel and review datasets, merging them for effective analysis.

### Key Sections
1. **Installing BM25:** Guides installation and setup of the BM25 library.
2. **Download Data:** Uses Kaggle API to fetch the necessary datasets.
3. **Data Preprocessing:** Handles cleaning and preparation of datasets, including:
   - **Hotels**: Metadata and attributes.
   - **Reviews**: Text reviews for recommendation modeling.
4. **Datasets Merging:** Combines datasets to create a unified input for modeling.
5. **Modeling:** Compares BM25 and a custom model for relevance scoring.

## Dependencies
The notebook uses the following Python libraries:
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical operations
- `re`: Regular expressions for text preprocessing
- `json`: Parsing and managing JSON data
- `rank_bm25`: BM25 algorithm for relevance ranking
- `kagglehub`: Accessing Kaggle datasets

### Installation
To install the required dependencies, run:
```bash
pip install pandas numpy rank_bm25 kagglehub
```

## Data Preprocessing
This document outlines the data preprocessing steps implemented for the **TripAdvisor Recommendation Challenge** project. The goal is to prepare the data for building and evaluating models, including a baseline BM25 model and a custom recommendation approach.

### Data Sources
Two datasets are used:
1. **Hotels Dataset (`offerings.csv`)**
2. **Reviews Dataset (`reviews.csv`)**

Download the dataset TripAdvisor Hotel Review from Kaggle:
[TripAdvisor Hotel Review Dataset](https://www.kaggle.com/datasets/joebeachcapital/hotel-reviews/data)

The notebook focuses on cleaning and structuring datasets for analysis:
- **Hotel Data:** Includes attributes such as location, amenities, and ratings.
- **Review Data:** Processes raw text reviews to remove noise and prepare inputs for the models.
- **Merging:** Combines the hotel and review datasets for a holistic analysis.

## Modeling
### BM25 Algorithm
The BM25 algorithm is implemented using the `rank_bm25` library to score the relevance of reviews to user queries. 

### Custom Model
A tailored model is developed to benchmark against BM25, leveraging additional features from the dataset.

## Results
The notebook provides insights into:
- How BM25 performs in recommending hotels based on reviews.
- The effectiveness of a custom-built model compared to BM25.

## How to Use
1. Clone the repository and navigate to the notebook directory:
   ```bash
   git clone https://github.com/BastienCherel/TripAdvisor-Recommendation-Challenge
   cd TripAdvisor-Recommendation-Challenge
   ```
2. Install the dependencies as listed above.
3. Open the notebook in Jupyter or Google Colab.
4. Run the cells sequentially to reproduce the results.

## Acknowledgments
This notebook is inspired by the TripAdvisor Recommendation Challenge. Special thanks to the authors of the `rank_bm25` library and contributors to Kaggle for providing datasets and tools.

## Authors
- [Bastien Cherel](https://github.com/BastienCherel)
- [Shangzhi Lou](https://github.com/ShangzhiLou)

---

Feel free to contribute to this project by submitting issues or pull requests!

