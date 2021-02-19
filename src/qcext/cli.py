import click 
import pkg_resources 

import qcext 
from qecsim.cli import _ConstructorParamType 

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
def speak(): 
	click.echo("Hello world!") 

@cli.command()
@_model_argument("code") 
@click.argument("time_steps", ) 
@click.argument("num_cycles", ) 
@_model_argument("error_model") 
@_model_argument("decoder") 
@_model_argument("readout") 
@click.argument("error_probabilities", )
@click.option("--max-failures", )
@click.option("--max-runs", ) 
@click.option("--measurement-error-probability", ) 
@click.option("--output", ) 
@click.option("--random-seed", )  
def run_ft(): 
	pass 

@cli.command() 
def merge(): 
	pass 