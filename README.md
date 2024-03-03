Joaquin Tejo\
Jorge Corro\
DDB Assignment IV

Using the provided financial phrasebank dataset, we preprocessed the sentences to create a corpus of text that could be used for training machine learning models. We vectorized the text data to generate numeric feature representations suitable for classification.

We trained several models, including Logistic Regression (LR), KNN, and Random Forest, on this dataset. To improve performance, we scaled and normalized the feature vectors so that regularization penalties would be applied evenly during training.

We evaluated the models by comparing their accuracy scores and confusion matrices. The Logistic Regression model achieved the best performance overall on this dataset.
Additionally, we fine-tuned a pretrained FinBERT NLP model on the same data. FinBERT outperformed Logistic Regression, with more balanced precision and recall across classes, especially for the neutral sentiment category.

To demonstrate real-world application, we built a Streamlit application that analyzes sentiment of financial news headlines from Yahoo Finance using our FinBERT model. The model shows strong performance, indicating the potential value of fine-tuned language models for financial sentiment analysis.
