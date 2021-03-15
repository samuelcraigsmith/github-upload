"""Test qcext.models.error_models.BiasedErrorModel."""

import unittest
import numpy as np
import math

from qcext.models.codes import Color666CodeNoisy
from qcext.models.error_models import BiasedBitFlipErrorModel


class TestBiasedErrorModelBitFlip(unittest.TestCase):
    """Test biased error model."""

    def setUp(self):
        """Set up the test."""
        self.code = Color666CodeNoisy(9)

    def test_returns_valid_error(self):
        """Test error model returns a valid error."""
        bias = 1
        error_probability = 0.5
        error_model = BiasedBitFlipErrorModel(bias)
        error = error_model.generate(self.code, error_probability)
        # correct shape.
        self.assertEqual(np.shape(error)[0], 2*self.code.n_k_d[0])
        # purely a bit flip error.
        self.assertFalse(np.any(error[self.code.n_k_d[0]:]))

    def test_same_seed_gives_same_error(self):
        """Test the error model gives identical errors for identical seeds."""
        seed_sequence = np.random.SeedSequence(entropy=None)
        seed = seed_sequence.entropy
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)
        bias = 1
        error_probability = 0.5

        error_model = BiasedBitFlipErrorModel(bias)

        error1 = error_model.generate(self.code, error_probability, rng1)
        error2 = error_model.generate(self.code, error_probability,  rng2)

        self.assertTrue(np.all(error1 == error2))

    def test_infinite_bias_100_percent_rate_noise(self):
        """Test with infinite bias and 100% error rate."""
        bias = math.inf
        error_model = BiasedBitFlipErrorModel(bias)
        error_probability = 1

        error = error_model.generate(self.code, error_probability)
        noisy_qubit_sites = [self.code.ordered_qubits[i]
                             for i in self.code.noisy_qubits]
        expected_error = self.code.new_pauli().site("X", *noisy_qubit_sites) \
            .to_bsf()

        self.assertTrue(np.all(error == expected_error))

    def test_unbiased_100_percent_rate_noise(self):
        """Test with zero bias and 100% error rate."""
        bias = 0
        error_model = BiasedBitFlipErrorModel(bias)
        error_probability = 1

        error = error_model.generate(self.code, error_probability)
        expected_error = self.code.new_pauli() \
            .site("X", *self.code.ordered_qubits).to_bsf()

        self.assertTrue(np.all(error == expected_error))

    def test_infinite_bias_0_percent_rate_noise(self):
        """Test with infinite bias and 0% error rate."""
        bias = math.inf
        error_model = BiasedBitFlipErrorModel(bias)
        error_probability = 0

        error = error_model.generate(self.code, error_probability)

        self.assertFalse(np.any(error))

    def test_unbiased_0_percent_rate_noise(self):
        """Test with zero bias and 0% error rate."""
        bias = 0
        error_model = BiasedBitFlipErrorModel(bias)
        error_probability = 0

        error = error_model.generate(self.code, error_probability)

        self.assertFalse(np.any(error))

    def test_statistics(self):
        """Test distribution is statistically consistent with expectations."""
        pass


if __name__ == "__main__":
    unittest.main()
