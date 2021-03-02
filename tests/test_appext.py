import unittest
from parameterized import parameterized
import logging
import numpy as np

from qcext.appext import run_ft
from qcext.models.codes import Color666CodeNoisy
from qcext.models.error_models import RestrictedDepolarizingErrorModel
from qcext.models.decoders import IsingDecoder
from qcext.models.readouts import Color666NoisyReadoutZ

logger = logging.getLogger(__name__)


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
    ])
    def test_run_ft(self, _, arguments):
        code, error_model, decoder, readout = arguments
        time_steps = 1
        n_cyc = 1
        error_probability = 0.1

        data = run_ft(code, time_steps, n_cyc, error_model, decoder, readout,
                      error_probability)

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

    


if __name__ == "__main__":
    unittest.main()
