from __future__ import annotations

from argparse import ArgumentParser

from . import config, expFromRnO

# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from types import ModuleType

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
    parser.add_argument('-pe', '--perform-experiment', action='store', type=int, nargs='+',
                        help="Used to set which experiments to run according to the paper.")

    return parser

def main() -> None:
    parser = prep_parser()
    known_args, _ = parser.parse_known_args()
    
    # Patch defaults for complex args
    try:
        known_args.perform_experiment = known_args.perform_experiment if known_args.perform_experiment else []
    except AttributeError:
        known_args.perform_experiment = []

    # Set configuration values
    # print(dir(known_args))
    config.DEBUG = known_args.debug
    config.ODOR_CONCENTRATION = known_args.odor_concentration
    config.PEAK_AFFINITY = known_args.peak_affinity
    config.MIN_AFFINITY = known_args.min_affinity
    config.HILL_COEFF = known_args.hill_coefficient
    config.ODOR_REPETITIONS = known_args.odor_repetitions
    config.ANGLES_REP = known_args.angle_reps

    print(f"Performing experiments... {','.join(map(str, known_args.perform_experiment))}")
    expFromRnO.test([expFromRnO.DEFAULT_EXPERIMENTS[i-1] for i in known_args.perform_experiment])

if __name__ == '__main__':
    main()