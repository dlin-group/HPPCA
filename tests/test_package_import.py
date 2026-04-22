def test_public_api_imports():
    import hppca

    assert hppca.__version__
    assert callable(hppca.fit_hppca)
    assert callable(hppca.fit_hppca_alg1)
    assert callable(hppca.fit_hppca_alg2_cs)
