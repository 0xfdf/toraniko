import pytest
import numpy as np
from toraniko.model import _factor_returns


@pytest.fixture
def sample_data():
    n_assets = 100
    n_sectors = 10
    n_styles = 5

    np.random.seed(42)
    returns = np.random.randn(n_assets, 1)
    mkt_caps = np.abs(np.random.randn(n_assets, 1))
    sector_scores = np.random.randint(0, 2, size=(n_assets, n_sectors))
    style_scores = np.random.randn(n_assets, n_styles)

    return returns, mkt_caps, sector_scores, style_scores


def test_output_shape_and_values(sample_data):
    returns, mkt_caps, sector_scores, style_scores = sample_data
    fac_ret, epsilon = _factor_returns(returns, mkt_caps, sector_scores, style_scores, True)

    assert fac_ret.shape == (1 + sector_scores.shape[1] + style_scores.shape[1], 1)
    assert epsilon.shape == returns.shape
    # evaluate expected vs actual fac_ret and epsilon as test vectors
    expected_fac_ret = np.array(
        [
            [-0.1619187],
            [0.0743045],
            [-0.21585373],
            [-0.27895516],
            [0.21496233],
            [-0.09829407],
            [-0.36415363],
            [0.16375184],
            [0.12933617],
            [0.35729484],
            [0.0176069],
            [0.13510884],
            [0.10831872],
            [-0.05781987],
            [0.11867375],
            [0.07687654],
        ]
    )
    np.testing.assert_array_almost_equal(fac_ret, expected_fac_ret)
    expected_epsilon_first_10 = np.array(
        [
            [0.04899388],
            [0.30702498],
            [0.23548647],
            [1.29181826],
            [0.30327413],
            [0.22804785],
            [1.40533523],
            [1.37903988],
            [0.08243883],
            [1.37698801],
        ]
    )
    np.testing.assert_array_almost_equal(epsilon[:10], expected_epsilon_first_10)


def test_residualize_styles(sample_data):
    returns, mkt_caps, sector_scores, style_scores = sample_data

    # if we residualize the styles we should obtain different returns out of the function

    fac_ret_res, _ = _factor_returns(returns, mkt_caps, sector_scores, style_scores, True)
    fac_ret_non_res, _ = _factor_returns(returns, mkt_caps, sector_scores, style_scores, False)

    assert not np.allclose(fac_ret_res, fac_ret_non_res)


def test_sector_constraint(sample_data):
    returns, mkt_caps, sector_scores, style_scores = sample_data
    fac_ret, _ = _factor_returns(returns, mkt_caps, sector_scores, style_scores, True)

    sector_returns = fac_ret[1 : sector_scores.shape[1] + 1]
    assert np.isclose(np.sum(sector_returns), 0, atol=1e-10)


def test_zero_returns():
    n_assets = 50
    n_sectors = 5
    n_styles = 3

    returns = np.zeros((n_assets, 1))
    mkt_caps = np.ones((n_assets, 1))
    sector_scores = np.random.randint(0, 2, size=(n_assets, n_sectors))
    style_scores = np.random.randn(n_assets, n_styles)

    fac_ret, epsilon = _factor_returns(returns, mkt_caps, sector_scores, style_scores, True)

    assert np.allclose(fac_ret, 0)
    assert np.allclose(epsilon, 0)


def test_market_cap_weighting(sample_data):
    returns, mkt_caps, sector_scores, style_scores = sample_data

    fac_ret1, _ = _factor_returns(returns, mkt_caps, sector_scores, style_scores, True)
    fac_ret2, _ = _factor_returns(returns, np.ones_like(mkt_caps), sector_scores, style_scores, True)

    assert not np.allclose(fac_ret1, fac_ret2)


def test_reproducibility(sample_data):
    returns, mkt_caps, sector_scores, style_scores = sample_data

    fac_ret1, epsilon1 = _factor_returns(returns, mkt_caps, sector_scores, style_scores, True)
    fac_ret2, epsilon2 = _factor_returns(returns, mkt_caps, sector_scores, style_scores, True)

    assert np.allclose(fac_ret1, fac_ret2)
    assert np.allclose(epsilon1, epsilon2)
