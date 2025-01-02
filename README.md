# Article Recommendation System and ChatPDF with AI

This project implements an article recommendation system.

## Data

The `data` directory contains the raw and processed data used in the project.

- `data.csv`: Contains the initial dataset.
- `filtered_data.csv`: Contains the processed and filtered dataset.

## Models

The `models` directory stores the trained models and related files.

- `embeddings/embeddings.pkl`: Contains the pre-computed embeddings for the articles.
- `sentences/sentences.pkl`: Contains the processed sentences from the articles.

## Notebooks

The `notebooks` directory contains Jupyter notebooks used for exploration, analysis, and model development.

- `0_arxiv_scraping.ipynb`: Notebook for scraping data from arXiv.
- `01_eda_and_preprocessing.ipynb`: Notebook for exploratory data analysis and preprocessing.
- `02_embedding.ipynb`: Notebook for generating article embeddings.
- `03_production_feature_pipeline.ipynb`: Notebook for the production feature pipeline.
- `04_production_embedding_pipeline.ipynb`: Notebook for the production embedding pipeline.

## Src

The `src` directory contains the Python source code for the project.

- `__init__.py`: Initializes the `src` directory as a Python package.
- `arxiv_scraper.py`: Contains the code for scraping articles from arXiv.
- `constants.py`: Defines project-wide constants.
- `streamlit_app.py`: Contains the code for the Streamlit web application.


## Data

The `data` directory contains the raw and processed data used in the project.

- `articles`: Contains the raw articles scraped from arXiv.
- `embeddings`: Contains the pre-computed embeddings for the articles.
- `sentences`: Contains the processed sentences from the articles.


## Streamlit

The `streamlit_app.py` file contains the code for the Streamlit web application.

## Usage

To run the Streamlit web application, execute the following command:

```
streamlit run streamlit_app.py
```

This will start the Streamlit server and open the web application in your default web browser.

## License

This project is licensed under the MIT License.