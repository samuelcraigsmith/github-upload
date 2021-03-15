import click
import pkg_resources
import json
import logging

import qcext
from qcext import appext
from qecsim.cli import _ConstructorParamType

logger = logging.getLogger(__name__)


def _model_argument(model_type): 
    # most important role of this function is to provide the dictionary to map strings to constructors. This looks at qcext entry points. 

    def _decorator(func):
        # extract name and class from entry-point, e.g. {'five_qubit': FiveQubitCode, ...}
        entry_point_id = 'qcext.cli.{}.{}s'.format(func.__name__, model_type)  # e.g. qecsim.cli.run_ftp.codes
        entry_points = sorted(pkg_resources.iter_entry_points(entry_point_id), key=lambda ep: ep.name)
        constructors = {ep.name: ep.load() for ep in entry_points}
        # add argument decorator
        func = click.argument(model_type, type=_ConstructorParamType(constructors), metavar=model_type.upper())(func)

        # TODO: add CLI support for qcext docstrings. 

        # # extract name and cli_help, e.g. [('five_qubit', '5-qubit'), ...]
        # model_definition_list = [(name, getattr(cls, ATTR_CLI_DESCRIPTION, '')) for name, cls in constructors.items()]
        # # update __doc__
        # formatter = click.HelpFormatter()
        # formatter.indent()
        # if model_definition_list:
        #     formatter.write_dl(model_definition_list)
        # model_doc_placeholder = '#{}_PARAMETERS#'.format(model_type.upper())  # e.g. #CODE_PARAMETERS#
        # func.__doc__ = inspect.getdoc(func).replace(model_doc_placeholder, formatter.getvalue())
        return func

    return _decorator


@click.group()
@click.version_option(version=qcext.__version__, prog_name='qcext')
def cli():
    pass


@cli.command()
@_model_argument("code")
@click.argument("time_steps", type=click.IntRange(min=1))
@click.argument("num_cycles", type=click.IntRange(min=1))
@_model_argument("error_model")
@_model_argument("decoder")
@_model_argument("readout")
@click.argument("error_probabilities", type=float)
@click.option("--max-failures", )
@click.option("--max-runs", "-r", type=click.IntRange(min=1), metavar="INT",)
@click.option("--measurement-error-probability", "-m", type=float, default=None)
@click.option("--output", "-o", default="-", type=click.Path(allow_dash=True))
@click.option("--random-seed", )
def run_ft(code, time_steps, num_cycles, error_model, decoder, readout,
           error_probabilities, max_failures, max_runs,
           measurement_error_probability, output, random_seed):
    # INPUT
    code.validate()

    # RUN
    runs_data = [appext.run_ft(code, time_steps, num_cycles, error_model,
                 decoder, readout, error_probabilities, max_failures=max_failures,
                 max_runs=max_runs, random_seed=random_seed,
                 measurement_error_probability=measurement_error_probability)]

    # OUTPUT
    _write_data(output, runs_data)


def _write_data(output, data):
    """ 
    Write data in JSON format (sorted keys) to the given output.
    Note: If the data cannot be written to the given output, for example if the file already exists, then the data is
    written to stderr and an exception is raised.
    :param output: Output file path or '-' for stdout.
    :type output: str
    :param data: Data (convertible to JSON).
    :type data: list of dict
    :raises ClickException: if the data cannot be written to the given path.
    """
    if output == '-':
        # write to stdout
        click.echo(json.dumps(data, sort_keys=True))
    else:
        try:
            # attempt to save to output filename (mode='x' -> fail if file exists)
            with open(output, 'x') as f:
                json.dump(data, f, sort_keys=True)
        except IOError as ex:
            logger.error('recovered data:\n{}'.format(json.dumps(data, sort_keys=True)))
            raise click.ClickException('{} (failed to open output file "{}")'.format(output, ex))