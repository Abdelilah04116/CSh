# Named Entity Recognition (NER) System for Review Analysis

## Overview
This repository contains a Jupyter Notebook (`CSh_NER.ipynb`) dedicated to building and utilizing a Named Entity Recognition (NER) system to analyze entities in a dataset of reviews. The system leverages the **spaCy** library with the pre-trained `en_core_web_sm` model to extract named entities such as organizations, locations, products, and persons from text data.

## Purpose
The notebook processes a dataset (`df_reviews.csv`) containing review texts and performs the following tasks:
- Extracts named entities from the reviews using spaCy's NER capabilities.
- Analyzes the distribution of entities across different sentiments (positive, negative, neutral).
- Visualizes the frequency of entity types per sentiment using bar plots.
- Provides a function to analyze new text inputs and display their extracted entities with visualizations.

## Dataset
The dataset (`df_reviews.csv`) is expected to have the following columns:
- `sentiment`: The sentiment of the review (positive, negative, or neutral).
- `text`: The original review text used for NER.
- `tokens`: Tokenized version of the text.
- `length`: Length of the text.
- `processed_text`: Preprocessed version of the text.

The notebook processes 50,000 review texts to extract and analyze entities.

## Dependencies
To run the notebook, ensure you have the following Python libraries installed:
- `pandas`
- `spacy` (with the `en_core_web_sm` model: `python -m spacy download en_core_web_sm`)
- `matplotlib`
- `seaborn`
- `collections`
- `warnings`

You can install them using:
```bash
pip install pandas spacy matplotlib seaborn
python -m spacy download en_core_web_sm
```

## Notebook Structure
1. **Library Imports**:
   - Loads necessary libraries and the spaCy `en_core_web_sm` model for NER.
2. **Data Loading**:
   - Reads the `df_reviews.csv` dataset and extracts the `text` column for analysis.
3. **Entity Extraction**:
   - Defines a function `extract_entities` to process texts and extract entities (e.g., ORG, GPE, PRODUCT, PERSON) with their labels, descriptions, and positions.
4. **Sentiment-Based Analysis**:
   - Groups entities by sentiment and displays the most frequent entities for key types (ORG, GPE, PRODUCT, PERSON).
   - Creates bar plots to visualize entity type frequencies per sentiment.
5. **New Text Analysis**:
   - Includes a function `analyze_new_text` to extract and categorize entities from user-provided text, with a visualization using spaCy's `displacy`.

## Key Features
- **Entity Extraction**: Identifies named entities in review texts, including their types and positions.
- **Sentiment Analysis**: Analyzes how entities vary across positive, negative, and neutral reviews.
- **Visualization**: Generates bar plots for entity type frequencies and visualizes entities in new texts.
- **Modular Code**: Provides reusable functions for entity extraction and analysis.

## Usage
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies (see above).
3. Place the `df_reviews.csv` dataset in the same directory as the notebook.
4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook CSh_NER.ipynb
   ```
5. Execute the cells sequentially to process the dataset and analyze entities.
6. Use the `analyze_new_text` function to test NER on custom text inputs.

## Example Output
For a sample text like:
> "I visited McDonald's in New York last week. The Big Mac was delicious and the staff at this location was very friendly."

The notebook outputs:
- Extracted entities (e.g., "McDonald's" as ORG, "New York" as GPE, "last week" as DATE).
- Categorized entities (e.g., Locations: New York, Organizations: McDonald's, The Big Mac).
- A visual representation of entities using `displacy`.

## Limitations
- The notebook uses the pre-trained `en_core_web_sm` model, which may not be optimized for specific domains or custom entity types.
- No model training or saving is performed; the analysis relies on the default spaCy model.
- Processing 50,000 texts can be computationally intensive and may require optimization for larger datasets.

## Future Improvements
- Train a custom spaCy model for domain-specific entities (e.g., restaurant-specific terms).
- Save extracted entities to a file (e.g., CSV or JSON) for further analysis.
- Optimize entity extraction for large datasets using batch processing or parallelization.

## Realized By
This project was realized by **Ourti Abdelilah** for the **Commonshare Task**.
