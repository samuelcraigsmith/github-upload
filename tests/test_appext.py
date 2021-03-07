import unittest
from parameterized import parameterized
import logging
import numpy as np

from qecsim.model import ErrorModel
from qcext.appext import run_ft, run_once_ft
from qcext.models.codes import Color666CodeNoisy
from qcext.models.error_models import RestrictedDepolarizingErrorModel, BiasedBitFlipErrorModel
from qcext.models.decoders import IsingDecoder, HybridDecoderPerfectMeasurement
from qcext.models.readouts import Color666NoisyReadoutZ
from qcext.models.readouts import Color666DoNothingReadoutZ


logger = logging.getLogger(__name__)


class _FixedErrorModel(ErrorModel):
    def __init__(self, errors):
        self.enumerate_errors = enumerate(errors)

    def generate(self, code, probability, rng=None):
        index, error  = next(self.enumerate_errors)
        logger.debug("Generating error {}".format(index))
        return error

    def probability_distribution(self, probability):
        pass

    @property
    def label(self):
        return 'fixed'


class TestRunFT(unittest.TestCase):
    @parameterized.expand([
        ["old_error_model_api", (
            Color666CodeNoisy(9),
            RestrictedDepolarizingErrorModel(
                Color666CodeNoisy(9).noisy_qubits
            ),
            IsingDecoder(),
            Color666NoisyReadoutZ(),
        )],
        ["new_error_model_api", (
            Color666CodeNoisy(9),
            RestrictedDepolarizingErrorModel(),
            IsingDecoder(),
            Color666NoisyReadoutZ(),
        )],
        ["", (
            Color666CodeNoisy(9),
            BiasedBitFlipErrorModel(1),
            HybridDecoderPerfectMeasurement("no_local"),
            Color666DoNothingReadoutZ(),
        )]
    ])
    def test_run_ft(self, _, arguments):
        code, error_model, decoder, readout = arguments
        time_steps = 2
        n_cyc = 2
        error_probability = 0.1
        meas_err_prob = 0.

        data = run_ft(code, time_steps, n_cyc, error_model, decoder, readout,
                      error_probability,
                      measurement_error_probability=meas_err_prob, )

        expected_key_cls = {  # cls == None for unused.
            'code': str,
            'n_k_d': tuple,
            'time_steps': int,
            'num_cycles': int,
            'error_model': str,
            'decoder': str,
            'readout': str,
            'error_probability': float,
            'measurement_error_probability': float,
            'n_run': int,
            'n_success': int,
            'n_fail': int,
            'n_logical_commutations': tuple,
            'custom_totals': None,
            'error_weight_total': int,
            'error_weight_pvar': float,
            'logical_failure_rate': float,
            'physical_error_rate': float,
            'wall_time': float,
            'seed': int,
            'version': float,
            'initial_error': None,
        }

        self.assertEqual(data.keys(), expected_key_cls.keys())
        for key, cls_ in expected_key_cls.items():
            self.assertTrue(
                cls_ is None
                or data[key] is None
                or type(data[key]) == cls_,
                "Expected type(data[{}]) == {} but instead got {}".format(
                    key, cls_, type(data[key])
                )
            )
        self.assertEqual(data["n_run"], 1)
        self.assertEqual(data["n_success"] + data["n_fail"], data["n_run"])
        self.assertTrue(data["n_fail"] >= 0)
        self.assertTrue(data["n_success"] >= 0)
        self.assertEqual(data["logical_failure_rate"],
                         data["n_fail"]/data["n_run"])


class TestRunOnceFT(unittest.TestCase):
    @parameterized.expand([
        (
            Color666CodeNoisy(9),
            3,
            1,
            BiasedBitFlipErrorModel(1),
            HybridDecoderPerfectMeasurement("no_local"),
            Color666DoNothingReadoutZ(),
            0.1,
            0,
        ),
    ])
    def test_run_once_ft(self, code, time_steps, n_cyc, error_model, decoder, readout, error_probability, measurement_error_probability):
        data = run_once_ft(code, time_steps, n_cyc, error_model, decoder, readout, error_probability, measurement_error_probability=measurement_error_probability)
        expected_key_cls = {
            "error_weight": None,
            "success": bool,
            "logical_commutations": None,
            "custom_values": None,
        }
        self.assertEqual(data.keys(), expected_key_cls.keys(), "data={} has missing/extra keys")
        for key, cls_ in expected_key_cls.items():
            assert data[key] is None or isinstance(data[key], cls_), "data[{}]={} is of type {} not of type={}.".format(key, data[key], type(data[key]), cls_)


class TestRunFTFails(unittest.TestCase):
    def setUp(self):
        self.code = Color666CodeNoisy(9)
        self.time_steps = 3
        self.n_cyc = 1
        self.errors = [
            self.code.new_pauli().to_bsf(),
            self.code.new_pauli().to_bsf(),
            self.code.logical_xs[0, :],
        ]
        self.decoder = HybridDecoderPerfectMeasurement("no_local")
        self.readout = Color666DoNothingReadoutZ()
        self.error_probability = 0

    def test_run_once_ft_fails(self):
        error_model = _FixedErrorModel(self.errors)
        data = run_once_ft(self.code, self.time_steps, self.n_cyc, error_model, self.decoder, self.readout, self.error_probability)
        self.assertFalse(data["success"])

    def test_run_ft_fails(self):
        max_runs = 5
        error_model = _FixedErrorModel(self.errors*max_runs)
        data = run_ft(self.code, self.time_steps, self.n_cyc, error_model, self.decoder, self.readout,
                      self.error_probability, max_runs=max_runs)
        self.assertEqual(data["n_fail"], max_runs, str(data))


if __name__ == "__main__":
    unittest.main()
