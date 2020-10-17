def test_model():
    with open("metrics.txt", 'r') as f:
        a = f.read()

    assert len(a) > 1