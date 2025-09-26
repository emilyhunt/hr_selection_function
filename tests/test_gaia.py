from hr_selection_function.gaia import (
    M10Estimator,
    MSubsampleEstimator,
    GaiaDensityEstimator,
)
import numpy as np
import healpy as hp


pixels_for_testing = [0, 57, 5893, 58932, 3, 382, 382, 159532, 9694, 196607]


def test_M10Estimator():
    model = M10Estimator()

    # Some values looked up manually from the GaiaUnlimited package
    test_values = [
        21.0278,
        20.9648,
        21.0118,
        20.9611,
        20.9353,
        20.9351,
        20.9351,
        21.0907,
        20.9455,
        21.0904,
    ]

    # Convert into ra/dec
    ra, dec = hp.pix2ang(2**7, pixels_for_testing, nest=True, lonlat=True)

    result = model(ra, dec)
    np.testing.assert_allclose(result, test_values)


def test_MSubsampleEstimator():
    model = MSubsampleEstimator()

    # Some values looked up manually from the dataframe
    test_values = [
        20.473399,
        20.460272,
        20.783859,
        20.615404,
        20.396479,
        20.648832,
        20.648832,
        20.727514,
        20.797851,
        20.266360,
    ]

    # Convert into ra/dec
    l, b = hp.pix2ang(2**7, pixels_for_testing, nest=True, lonlat=True)  # noqa: E741

    result = model(l, b)
    np.testing.assert_allclose(result, test_values)


def test_GaiaDensityEstimator():
    l, b, pmra, pmdec, parallax, result_expected = _get_test_density_data()  # noqa: E741

    model = GaiaDensityEstimator()
    result_actual = model(l, b, pmra, pmdec, parallax)

    np.testing.assert_allclose(result_actual, result_expected)


def _get_test_density_data():
    return [
        [
            195.44967469843573,
            172.28930309346387,
            104.75874782403532,
            173.70832020346367,
            79.20595430423523,
            153.9832103408225,
            76.36865570717138,
            75.91799679118454,
            269.8536005362042,
            191.86293977733843,
        ],
        [
            -0.3109413445191218,
            -0.09090826566516344,
            -1.970488037640793,
            -2.534745624741465,
            1.975546326136809,
            -9.477611780892138,
            -0.36600912016968506,
            2.33643211648706,
            -2.839079251269452,
            -8.251482667198461,
        ],
        [
            0.41893829079702133,
            0.45112952571743464,
            -0.8864160229048276,
            -0.00944812985700761,
            -1.945501466102864,
            -0.2501085125841758,
            -1.1535825632274828,
            -1.9011820856534454,
            -5.555678886414437,
            1.0248139396694183,
        ],
        [
            0.5954508838720914,
            -0.5945587927019814,
            -2.2983623730571368,
            -1.2255169778814685,
            -3.1292798157719655,
            -0.4026029905131187,
            -2.047954786200685,
            -3.2077251711446126,
            5.1387683882097965,
            0.23107733843403835,
        ],
        [
            0.11226267600669465,
            0.09336113589903527,
            0.1087746540178971,
            0.11248267295854439,
            0.07306373461755672,
            0.14866912811593022,
            0.09964444948414934,
            0.09891577538495654,
            0.16250218203179784,
            0.343948431873036,
        ],
        [
            26670.688142002928,
            32035.043148849825,
            77662.2672329491,
            32962.30258016591,
            5788.030329350739,
            22917.85569373121,
            3136.150415190452,
            127494.21366708462,
            82293.64350935556,
            23629.392112275982,
        ],
    ]
