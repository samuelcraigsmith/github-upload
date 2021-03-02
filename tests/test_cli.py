import unittest
from click.testing import CliRunner
from parameterized import parameterized

from qcext.cli import cli


class TestCLI(unittest.TestCase):
    @parameterized.expand(["", "--help", "--version"])
    def test_cli(self, argument):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, argument)
            assert result.exit_code == 0

    @parameterized.expand([
        ("a_test", [
            "color666noisy(9)",
            "restricted_depolarizing",
            "ising_decoder",
            "color666noisy_z",
        ]),
    ])
    def test_cli_run_ft(self, _, arguments):
        time_steps = "1"
        num_cycles = "1"
        error_probability = "0.1"
        code, error_model, decoder, readout = arguments
        arguments = ["run-ft", code, time_steps, num_cycles, error_model,
                     decoder, readout, error_probability]
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, arguments)
            self.assertTrue(result.exit_code == 0)
        #  TODO: check returns data that is correctly formatted.


if __name__ == "__main__":
    unittest.main()
