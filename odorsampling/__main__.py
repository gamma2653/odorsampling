from __future__ import annotations

import sys
from argparse import ArgumentParser
from pprint import pprint

import yaml
import matplotlib

from . import config, experiments, testLayers, testRnO

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Iterable, Union


def parse_range(s: str) -> Iterable[int]:
    """
    Parses a range of numbers into a list of ints. (Inclusive)
    """
    if '-' not in s:
        raise ValueError(f"Invalid range: '{s}'.")
    start, end = map(int, map(lambda s: s.strip(), s.split('-')))
    return range(start, end+1)

def parse_list(s: str) -> Iterable[int]:
    """
    Parses a list of numbers into a list of ints.
    """
    if ',' not in s:
        raise ValueError(f"Invalid list: '{s}'.")
    return map(int, map(lambda s: s.strip(), s.split(',')))

def parse_range_or_int(s: str) -> Iterable[int]:
    try:
        return list(parse_range(s))
    except ValueError:
        try:
            return list(parse_list(s))
        except ValueError:
            return [int(s.strip())]


def prep_parser() -> ArgumentParser:
    """
    Preps argument parser.
    """
    parser = ArgumentParser(prog=config.NAME,
                            description=config.DESCRIPTION)
    parser.add_argument('-d', '--debug', action='store_true',
                        help="Set to enable more extensive logging and some other debugging flags.")
    parser.add_argument('-c', '--config', action='store', type=str, default="./experiment.yaml",
                        help="Experiment YAML file to load. See examples/experiments.yaml for an example.")


    parser.add_argument('-oc', '--odor-concentration', action='store', type=float,
                        help="Use to set the odor concentration. Scientific notation is allowed.")
    parser.add_argument('-pa', '--peak-affinity', action='store', type=float,
                        help="Used to set the peak affinity, scientific notation is allowed.")
    parser.add_argument('-ma', '--min-affinity', action='store', type=float,
                        help="Used to set the minimum affinity, scientific notation is allowed.")
    parser.add_argument('-hc', '--hill-coefficient', action='store', type=float,
                        help="Used to set the Hill Coefficient, scientific notation is allowed.")
    # TODO: Probably set this to an integer. Investigate this param.
    parser.add_argument('-or', '--odor-repetitions', action='store', type=float,
                        help="Used to set the odor repetitions.")
    # TODO: Understand this param better.
    parser.add_argument('-ar', '--angle-reps', action='store', type=float,
                        help="Used to set the angle reps, scientific notation is allowed.")
    # TODO: Better document
    parser.add_argument('-pe', '--perform-experiment', action='store', type=str, nargs='*', metavar='EXPERIMENT_NAME',
                        help="Used to set which experiments to run according to the paper. Can be defined as a single int, range, "
                        "or a list. Eg) '2' and '1-3' and '1,2,3' are all valid. If there are spaces, the argument must be in quotes.")
    parser.add_argument('-t', '--run-tests', action='store', type=str, nargs='*', metavar='TEST_NAME',
                        help="Used to set which tests to run. Eg) 'layers' and 'RnO' are valid.")
    parser.add_argument('-mplb', '--mpl-backend', action='store', type=str, default='tkAgg',
                        help="Used to set the backend used by matplotlib for the graphs.")
    parser.add_argument('-rs', '--random-seed', action='store', type=int, nargs='+', default=None,
                        help="Used to set the RNG's seed value. None/default operating system entropy used if not set."
                        "Eg) `python -m odorsampling -rs 1865`")

    return parser

def _special_init(namespace, name, init_factory, *args, **kwargs):
    """
    Falsey/DNE catch-all initialization
    """
    try:
        if not getattr(namespace, name):
            raise AttributeError
    except AttributeError:
        setattr(namespace, name, init_factory(*args, **kwargs))
        
#     if not getattr(namespace, name, None):
#         setattr(namespace, name, init_factory())

TESTS = {
    'layers': [testLayers.test],
    'RnO': [testRnO.test]
}


def perform_experiments(exps):
    print(f"Performing experiments... {','.join(map(str, exps))}")
    print(list(exps))
    experiments.test([experiments.DEFAULT_EXPERIMENTS[i-1] for i in exps])

def perform_tests(test_names: Iterable[str]):
    print(f"Performing tests... {','.join(map(str, test_names))}")
    for test in test_names:
        if test in TESTS:
            print(f"Performing test `{test}`.")
            [subtest() for subtest in TESTS[test]]

def main() -> None:
    parser = prep_parser()

    # Resolve configs
    # CMD line args
    known_args, _ = parser.parse_known_args()
    # Patch defaults for complex args (may no longer be necessary after some changes)
    _special_init(known_args, 'perform_experiment', list)
    _special_init(known_args, 'run_tests', list)

    # Matplotlib backend
    utils.set_seed(known_args.random_seed)
    try:
        matplotlib.use(known_args.mpl_backend)
    except ModuleNotFoundError as e:
        print(f"Unable to load matplotlib backend '{known_args.mpl_backend}'. Please ensure any required packages are installed to use this backend.")
        print(e, file=sys.stderr)
    print(f"Using {matplotlib.get_backend()} as the matplotlib backend.")

    # YAML config
    yaml_config: dict = {
        'parameters': {}
    }
    try:
        with open(known_args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            print('yaml')
            pprint(yaml_config)
            print('end yaml')
    except FileNotFoundError:
        print(f"Unable to load experiment file '{known_args.config}'.")
        sys.exit(1)
    
    # Override YAML config with command line args
    try:
        # Update config with only non-None values
        yaml_config['parameters'].update(((k,v) for k, v in known_args.__dict__.items() if v is not None))
    except KeyError | AttributeError:
        yaml_config['parameters'] = known_args.__dict__

    # Override python file config with YAML/CMD config
    for key, value in yaml_config['parameters'].items():
        if value is not None:
            setattr(config, key.upper(), value)
        
    print("Configuration:")
    pprint({ key: getattr(config, key) for key in dir(config) if not key.startswith('__')})

    perform_experiments(known_args.perform_experiment)
    perform_tests(known_args.run_tests)





if __name__ == '__main__':
    main()