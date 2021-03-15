import unittest
from click.testing import CliRunner
from parameterized import parameterized
import json

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
        ("a_test", [
            "color666noisy(9)",
            "biased_bit_flip(1)",
            "hybrid_decoder_perfect_measurement('local')",
            "color666_do_nothing_readout_z",
        ]),
        # outuput file
    ])
    def test_cli_model_arguments_run_ft(self, _, arguments):
        time_steps = "3"
        num_cycles = "3"
        error_probability = "0.1"
        measurement_error_probability = "-m0"
        n_runs = "-r10"
        code, error_model, decoder, readout = arguments
        arguments = ["run-ft", n_runs, measurement_error_probability, code, time_steps, num_cycles, error_model,
                     decoder, readout, error_probability]
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, arguments)
            print(result.output)
            self.assertTrue(result.exit_code == 0, result.output)
        #  TODO: check returns data that is correctly formatted.

    @parameterized.expand([
        #  output file
        ("output_file", [
            "-otmp_data.json", "-m0", "color666noisy(9)", "3", "3",
            "biased_bit_flip(1)", "hybrid_decoder_perfect_measurement('local')",
            "color666_do_nothing_readout_z", "0.1"
        ])
    ])
    def test_cli_run_ft(self, _, arguments):
        runner = CliRunner()
        with runner.isolated_filesystem():
            arguments = ["run-ft"] + arguments
            result = runner.invoke(cli, arguments)
            self.assertTrue(result.exit_code == 0, result.output)

    def test_cli_run_ft_existing_output_file(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            # run to existing file
            empty_file_name = "tmp_empty.json"
            with open(empty_file_name, "w") as f:
                result = runner.invoke(cli, [
                    "run-ft", "-o", empty_file_name, "color666noisy(9)",
                    "3", "3", "biased_bit_flip(1)",
                    "hybrid_decoder_perfect_measurement('local')",
                    "color666_do_nothing_readout_z", "0.1"
                ])
                # check file is still empty
                f.seek(0, 2)  # seek to end of file
                assert f.tell() == 0  # end of file is at position zero
                assert result.exit_code != 0

if __name__ == "__main__":
    unittest.main()
