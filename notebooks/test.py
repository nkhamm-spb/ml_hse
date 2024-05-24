import os
import pytest
import pandas as pd
from click.testing import CliRunner
from main import*


data_path = "../data/singapore_airlines_reviews.csv"
model_path = "test_model.pkl"
test_path = "../data/singapore_airlines_reviews.csv"

def test_train():
    runner = CliRunner()
    result = runner.invoke(train, ['--data', data_path, '--model', model_path, '--split', '0.3'])    
    assert result.exit_code == 0
    assert os.path.exists(model_path)
    os.remove(model_path)

def test_train_test_file():
    runner = CliRunner()
    result = runner.invoke(train, ['--data', data_path, '--model', model_path, '--test', test_path])    
    assert result.exit_code == 0
    assert os.path.exists(model_path)
    os.remove(model_path)

def test_predict_csv():
    runner = CliRunner()
    result = runner.invoke(train, ['--data', data_path, '--model', model_path, '--split', '0.3'])
    assert result.exit_code == 0
    result = runner.invoke(predict, ['--model', model_path, '--data', data_path])
    assert result.exit_code == 0
    assert result.output.count('\n') == pd.read_csv(data_path).shape[0]
    os.remove(model_path)

def test_predict_string():
    runner = CliRunner()
    result = runner.invoke(train, ['--data', data_path, '--model', model_path, '--split', '0.3'])
    assert result.exit_code == 0
    
    input_text = "It was the worst flight of my life. The flight was greatly delayed"
    result = runner.invoke(predict, ['--model', model_path, '--data', input_text])
    assert result.exit_code == 0
    assert int(result.output.strip()[1]) == 1
    
    input_text = "Great flight, great service. We arrived right on time"
    result = runner.invoke(predict, ['--model', model_path, '--data', input_text])
    assert result.exit_code == 0
    assert int(result.output.strip()[1]) == 5
    os.remove(model_path)
    
if __name__ == "__main__":
    pytest.main()