import re
import spacy
import click
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pickle


def text_process(text, nlp):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\n', ' ', text)
    doc = nlp(text)
    filtered_words = [token.lemma_ for token in doc if not token.is_stop]
    clean_text = ' '.join(filtered_words)
    return clean_text

@click.group()
def main():
    pass
    
@main.command()
@click.option('--data', type=str, help='Путь к датасету для обучения', required=True)
@click.option('--test', type=str, help='Путь к датасету для тестирования', default=None)
@click.option('--split', type=float, help='Доля тестовых данных', default = None)
@click.option('--model', type=str, help='Путь для сохранения модели', required=True)
def train(data, test, split, model):
    df = pd.read_csv(data)
    vectorizer = TfidfVectorizer()
    
    nlp = spacy.load("en_core_web_sm")
    text_processed = df['text'].map(lambda text: text_process(text, nlp))
    
    X_train, X_test, y_train, y_test = text_processed, None, df['rating'], None

    if split != None:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=split, random_state=42)
    elif test != None:
        df_test = pd.read_csv(test)
        X_test = df_test['text'].map(lambda text: text_process(text, nlp))
        y_test = df_test['rating']
    else: 
        raise AssertionError("No test data")

    X_train = vectorizer.fit_transform(X_train)
    clf_logistic = LogisticRegression().fit(X_train, y_train)
    
    X_test = vectorizer.transform(X_test)
    y_pred = clf_logistic.predict(X_test)
    f1 = f1_score(y_pred, y_test, average='micro')
    click.echo(f"F1-мера на тестовой выборке: {f1}")
    
    with open(model, 'wb') as file:
        pickle.dump((clf_logistic, vectorizer), file)


@main.command()
@click.option('--model', type=str, help='Путь к обученной модели', required=True)
@click.option('--data', type=str, help='Текст для предсказания', required=True)
def predict(model, data):
    with open(model, 'rb') as file:
        clf_logistic, vectorizer = pickle.load(file)
    nlp = spacy.load("en_core_web_sm")
    
    if data.endswith(".csv"):
        df = pd.read_csv(data)
        text_processed = df['text'].map(lambda text: text_process(text, nlp))
        
        text_processed = vectorizer.transform(text_processed)
        prediction = clf_logistic.predict(text_processed)
        for p in prediction:
            click.echo(p)
    
    else:
        text = pd.Series(data).map(lambda text: text_process(text, nlp))
        new_test = vectorizer.transform(text)

        prediction = clf_logistic.predict(new_test)
        click.echo(prediction)

if __name__ == '__main__':
    main()
