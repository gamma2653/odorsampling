from __future__ import annotations

import sys
from argparse import ArgumentParser
from pprint import pprint


import matplotlib

from . import config, expFromRnO, testLayers, testRnO

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Iterable

def prep_parser() -> ArgumentParser:
    """
    Preps argument parser.
    """
    parser = ArgumentParser(prog=config.NAME,
                            description=config.DESCRIPTION)
    parser.add_argument('-d', '--debug', action='store_true', default=config.DEBUG,
                        help="Set to enable more extensive logging and some other debugging flags.")
    parser.add_argument('-oc', '--odor-concentration', action='store', type=float, default=config.ODOR_CONCENTRATION,
                        help="Use to set the odor concentration. Scientific notation is allowed.")
    parser.add_argument('-pa', '--peak-affinity', action='store', type=float, default=config.PEAK_AFFINITY,
                        help="Used to set the peak affinity, scientific notation is allowed.")
    parser.add_argument('-ma', '--min-affinity', action='store', type=float, default=config.MIN_AFFINITY,
                        help="Used to set the minimum affinity, scientific notation is allowed.")
    parser.add_argument('-hc', '--hill-coefficient', action='store', type=float, default=config.HILL_COEFF,
                        help="Used to set the Hill Coefficient, scientific notation is allowed.")
    # TODO: Probably set this to an integer. Investigate this param.
    parser.add_argument('-or', '--odor-repetitions', action='store', type=float, default=config.ODOR_REPETITIONS,
                        help="Used to set the odor repetitions.")
    # TODO: Understand this param better.
    parser.add_argument('-ar', '--angle-reps', action='store', type=float, default=config.ANGLES_REP,
                        help="Used to set the angle reps, scientific notation is allowed.")
    # TODO: Better document
    parser.add_argument('-pe', '--perform-experiment', action='store', type=int, nargs='*', metavar='EXP#',
                        help="Used to set which experiments to run according to the paper.")
    parser.add_argument('-t', '--run-tests', action='store', type=str, nargs='*', metavar='TEST_NAME',
                        help="Used to set which tests to run. Eg) 'layers' and 'RnO' are valid.")
    parser.add_argument('-mplb', '--mpl-backend', action='store', type=str, default='tkAgg',
                        help="Used to set the backend used by matplotlib for the graphs.")

    return parser

def _special_init(namespace, name, init_factory):
    try:
        if not getattr(namespace, name):
            raise AttributeError
    except AttributeError:
        setattr(namespace, name, init_factory())
        
#     if not getattr(namespace, name, None):
#         setattr(namespace, name, init_factory())

TESTS = {
    'layers': [testLayers.test],
    'RnO': [testRnO.test]
}

def perform_experiments(experiments):
    print(f"Performing experiments... {','.join(map(str, experiments))}")
    expFromRnO.test([expFromRnO.DEFAULT_EXPERIMENTS[i-1] for i in experiments])

def perform_tests(test_names: Iterable[str]):
    print(f"Performing tests... {','.join(map(str, test_names))}")
    for test in test_names:
        if test in TESTS:
            print(f"Performing test `{test}`.")
            [subtest() for subtest in TESTS[test]]

def main() -> None:
    parser = prep_parser()
    known_args, _ = parser.parse_known_args()
    
    # Patch defaults for complex args
    _special_init(known_args, 'perform_experiment', list)
    _special_init(known_args, 'run_tests', list)
    print("Configuration:")
    pprint(known_args.__dict__)
    # Set configuration values
    # print(dir(known_args))
    config.DEBUG = known_args.debug
    config.ODOR_CONCENTRATION = known_args.odor_concentration
    config.PEAK_AFFINITY = known_args.peak_affinity
    config.MIN_AFFINITY = known_args.min_affinity
    config.HILL_COEFF = known_args.hill_coefficient
    config.ODOR_REPETITIONS = known_args.odor_repetitions
    config.ANGLES_REP = known_args.angle_reps
    try:
        matplotlib.use(known_args.mpl_backend)
    except ModuleNotFoundError as e:
        print(f"Unable to load matplotlib backend '{known_args.mpl_backend}'. Please ensure any required packages are installed to use this backend.")
        print(e, file=sys.stderr)
    print(f"Using {matplotlib.get_backend()} as the matplotlib backend.")
    perform_experiments(known_args.perform_experiment)
    perform_tests(known_args.run_tests)





if __name__ == '__main__':
    main()