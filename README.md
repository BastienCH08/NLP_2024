
# Natural Language Processing : TripAdvisor Recommendation Challenge 

## Beating BM25 

The objective of this challenge is to create a unique recommendation system for TripAdvisor reviews. This involves utilizing a range of pre-treatments and vocabulary manipulation techniques on the reviews. Both pre-trained machine learning models and custom-developed models will be employed as needed. By integrating various methodologies, this approach aims to optimize the recommendation process while adhering to the constraint of not using direct supervised learning to predict the best place based on a specific query.

## Data Preprocessing 

### Overview
This document outlines the data preprocessing steps implemented for the **TripAdvisor Recommendation Challenge** project. The goal is to prepare the data for building and evaluating models, including a baseline BM25 model and a custom recommendation approach.

---

### Data Sources
Two datasets are used:
1. **Hotels Dataset (`offerings.csv`)**
2. **Reviews Dataset (`reviews.csv`)**

Download the dataset TripAdvisor Hotel Review from Kaggle :
[TripAdvisor Hotel Review Dataset](https://www.kaggle.com/datasets/joebeachcapital/hotel-reviews/data)

---

### Steps Performed

#### 1. Loading Datasets
- Both datasets are loaded using the `pandas` library.
- `data_hotels` and `data_reviews` are created to hold the respective data.

```python
import pandas as pd

data_hotels = pd.read_csv('../offerings.csv')
data_reviews = pd.read_csv('../reviews.csv')
```

---

#### 2. Dropping Irrelevant Columns

##### Reviews Dataset
Columns not directly relevant for the modeling task are removed:
- Dropped columns: `id`, `via_mobile`, `author`, `date`, `date_stayed`, `num_helpful_votes`, `title`.

```python
data_reviews = data_reviews.drop(columns=["id", "via_mobile", "author", "date", "date_stayed", "num_helpful_votes", "title"])
```

##### Hotels Dataset
Columns with minimal relevance or redundant information are excluded:
- Dropped columns: `phone`, `details`, `region_id`, `type`, `url`.

```python
data_hotels = data_hotels.drop(columns=["phone", "details", "region_id", "type", "url"])
```

---

#### 3. Parsing JSON-Like Strings

##### Reviews Dataset
- The `ratings` column in `data_reviews` contained JSON-like strings.
- A function, `convert_review_string`, is written to:
  - Fix malformed JSON strings using regular expressions.
  - Convert the strings to Python dictionaries.

- The parsed `ratings` data is normalized into separate columns and merged back into the dataset.
- Dropped irrelevant rating components such as `check_in_front_desk` and `business_service_(e_g_internet_access)`.

```python
import re
import json

def fix_json_format(address_string):
    address_string = re.sub(r"'", r'"', address_string)
    return address_string

def convert_review_string(address_string):
    if isinstance(address_string, str):
        fixed_string = fix_json_format(address_string)
        return json.loads(fixed_string)
    return address_string

data_reviews['ratings'] = data_reviews['ratings'].apply(convert_review_string)
reviews_df = pd.json_normalize(data_reviews['ratings'])
data_reviews = pd.concat([data_reviews, reviews_df], axis=1)
data_reviews.drop(columns=['ratings', 'check_in_front_desk', 'business_service_(e_g_internet_access)'], inplace=True)
```

##### Hotels Dataset
- Similarly, the `address` column in `data_hotels` contained JSON-like strings.
- A function, `convert_address_string`, is applied to parse and normalize the data:
  - Fixed common JSON formatting issues.
  - Extracted address components (e.g., `street`, `city`, `country`) into separate columns.
- Dropped the original `address` column post-normalization.

```python
def fix_json_format(address_string):
    address_string = re.sub(r"'", r'"', address_string)
    address_string = re.sub(r'("(?:[^"]|\\.)*?)\'(.*?")', r'\1\2', address_string)
    return address_string

def convert_address_string(address_string):
    if isinstance(address_string, str):
        fixed_string = fix_json_format(address_string)
        return json.loads(fixed_string)
    return address_string

# Apply the transformation
data_hotels['address'] = data_hotels['address'].apply(convert_address_string)
address_df = pd.json_normalize(data_hotels['address'])
data_hotels = pd.concat([data_hotels, address_df], axis=1)
data_hotels.drop(columns=['address'], inplace=True)
```

---

#### 4. Calculating Aggregate Ratings

- Created a new column `rating` in `data_reviews` to store the mean value of multiple rating components:
  - Included: `service`, `cleanliness`, `overall`, `value`, `location`, `sleep_quality`, `rooms`.
- After aggregation, the individual rating columns are dropped to reduce redundancy.

```python
import numpy as np

data_reviews["rating"] = data_reviews[[
    "service", "cleanliness", "overall", "value",
    "location", "sleep_quality", "rooms"
]].mean(axis=1)

data_reviews.drop(columns=["service", "cleanliness", "overall", "value", "location", "sleep_quality", "rooms"], inplace=True)
```

---

#### 5. Merging Datasets

- The `data_reviews` and `data_hotels` datasets are merged on the common key (`offering_id` in reviews and `id` in hotels).
- To avoid name collisions, a prefix is added to columns:
  - Columns from `data_hotels`: `hotel_`
  - Columns from `data_reviews`: `reviews_`
- Rows with missing values are dropped.

```python
# Add prefixes to column names for clarity
hotel_columns = {col: f'hotel_{col}' for col in data_hotels.columns if col != 'offering_id'}
reviews_columns = {col: f'reviews_{col}' for col in data_reviews.columns if col != 'id'}

data = pd.merge(data_reviews, data_hotels, left_on="offering_id", right_on="id", how="left", suffixes=("_review", "_hotel"))
data = data.drop(columns=["offering_id"])
data = data.rename(columns={**hotel_columns, **reviews_columns})
data = data.dropna()
```

---

#### 6. Final Inspection

- Displayed the first few rows and dataset information to verify preprocessing.

```python
# Display processed data
print(data.head())
print(data.info())
```

---

### Outcome
- The preprocessing pipeline transformed raw data into a structured format suitable for analysis and modeling.
- Key transformations included:
  - Normalizing JSON-like fields into columns.
  - Aggregating and cleaning ratings.
  - Merging datasets while ensuring clear column naming.
- The resulting dataset is ready for model implementation and evaluation.

## BM25 vs. custom model


## Authors
- [Bastien Cherel](https://github.com/BastienCherel)
- [Shangzhi  Lou](https://github.com/ShangzhiLou)

 
