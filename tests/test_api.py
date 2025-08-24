import json
from fastapi.testclient import TestClient
from main import app

def test_root():
    client = TestClient(app)
    r = client.get('/')
    assert r.status_code == 200
    assert 'message' in r.json()


def test_predict():
    client = TestClient(app)
    payload = {"review": "This movie was absolutely wonderful and thrilling"}
    r = client.post('/predict', json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data['review'] == payload['review']
    assert data['sentiment'] in ["Positive", "Negative"]


def test_predict_empty_review():
    client = TestClient(app)
    payload = {"review": "   "}
    r = client.post('/predict', json=payload)
    assert r.status_code == 422


def test_model_info():
    client = TestClient(app)
    r = client.get('/model/info')
    assert r.status_code == 200
    data = r.json()
    assert data['model_name'] == 'sentiment-model'
    assert 'run_id' in data


def test_batch_predict():
    client = TestClient(app)
    payload = {"reviews": ["Great movie", "Terrible plot"]}
    r = client.post('/predict/batch', json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data['count'] == 2
    assert len(data['sentiments']) == 2
    for s in data['sentiments']:
        assert s in ["Positive", "Negative"]


def test_batch_predict_too_large():
    client = TestClient(app)
    payload = {"reviews": ["ok"] * 101}
    r = client.post('/predict/batch', json=payload)
    assert r.status_code == 422


def test_review_too_long():
    client = TestClient(app)
    long_text = "a" * 6000
    r = client.post('/predict', json={"review": long_text})
    assert r.status_code == 422
