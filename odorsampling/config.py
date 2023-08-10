import logging
import builtins


NAME = 'odorsampling'
DESCRIPTION = (f'This program allows for the user to define different configurations of a theoretical model '
               f'of odor sampling in the olfactory bulb. '
               f'')

LOG_MSG_FMT: str = '[%(asctime)s] [%(name)s]: [%(levelname)s] %(message)s'
LOG_DATE_FMT: str = '%m-%d-%Y %H:%M:%S'
LOG_FILE_NAME: str = 'output.log'

LOG_FORMATTER = logging.Formatter(LOG_MSG_FMT, LOG_DATE_FMT)

# Default Arguments
DEBUG = builtins.__debug__
LOG_LEVEL = logging.DEBUG if DEBUG else logging.WARNING
ODOR_CONCENTRATION = 1e-8
PEAK_AFFINITY = -8     # literally 10e-8, not influenced by minimum_affinity value
MIN_AFFINITY = 2   # asymptotic affinity exponent, negligible
HILL_COEFF = 1 # Hill Coefficient, often labeled as "m"
ODOR_REPETITIONS = 2 #Amount of odorscene repetitions to create a smooth graph
ANGLES_REP = 2

# location distributions control params, eg., uniform, gaussian...1 and only 1 type needs to be true at any given time
DIST_TYPE_GAUSS = False
DIST_TYPE_UNIF = True
MU = 6
SIG = 12


def default_log_setup(logger: logging.Logger, log_level: int = None, stream_handler_level = logging.WARNING, file_handler_level = logging.DEBUG):
    """
    Automatically adds LOG_FILE_HANDLER and LOG_STREAM_HANDLER as handlers.

    Parameters
    ----------
    logger -
        The logger to setup with the default configuration.
    """
    logger.setLevel(LOG_LEVEL if log_level is None else log_level)
    file_handler = logging.FileHandler(LOG_FILE_NAME)
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(LOG_FORMATTER)
    stream_handler.setFormatter(LOG_FORMATTER)
    file_handler.setLevel(file_handler_level)
    stream_handler.setLevel(stream_handler_level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)




GL_EXT = ".gl"
"""
Extension used by saved glom cell layers.
"""
ML_EXT = ".ml"
"""
Extension used by saved mitral cell layers.
"""
MCL_MAP_EXT = ".glml_map"

# parameters for odor/recepter coverage ellipse graph
RECEPTOR_ELLIPSE_STANDARD_DEVIATION = 1.5
ODORSCENE_INDEX = 12 # e.g., 0 - 27 odorscenes
ODORSCENE_REP_NUMBER = 1 # e.g., 0 - 199 repetitions
ODOR_COLOR = 'black'
# RECEPTOR_INDEX = 21 # e.g., pick a receptor from 0 - 29
RECEPTOR_INDEX = 'ALL' # e.g., display receptors

SHOW_SDA_ELLIPSE = True
SHOW_SDE_ELLIPSE = True

SDA_COLOR = 'red'
SDE_COLOR = 'blue'
SDA_FILL = False
SDE_FILL = False

LINE_WIDTH = 2

GRAPH_TITLE = 'Receptor-Odor Coverage With STD=' + str(RECEPTOR_ELLIPSE_STANDARD_DEVIATION)
XLABEL = ''
YLABEL = ''
GRAPH_FILE_NAME = 'Receptor-Odor Coverage '



# Mock qspace
MOCK_QSPACE_DIMENSION = [0, 4]

# Use mock odor even when running calcs (not running the tool standalone)
USE_MOCK_ODORS_EVEN_WHEN_RUNNING_CALCS = False
# Mock odors
# MOCK_ODORS_X = [2, 3, 6, 8]
# MOCK_ODORS_Y = [4, 1, 7, 5]

MOCK_ODORS = [[2, 3.7], [3, 1], [2.3, 1.5], [3.8, 2.4], [2.7, 1.4], [2.9, 1.7], [1.9,2.567], [1.34, 2.43], [1.1, 1.4], [3.7,3.5]] # e.g., this mocks 4 odors with loc coordinates

# Use mock receptor even when running calcs (not running the tool standalone)
USE_MOCK_RECEPTORS_EVEN_WHEN_RUNNING_CALCS = False
# Mock receptors
# MOCK_RECEPTOR_MEAN = [1.96854058709, 2.46221881356]
MOCK_RECEPTOR_MEAN = [3.26854058709, 2.96221881356]
MOCK_RECEPTOR_SDA = [0.351717212085, 0.119741731921]
MOCK_RECEPTOR_SDE = [0.360607106742, 0.65383818835]


MOCK_RECEPTOR_MEAN1 = [1.48854058709, 1.66221881356]
MOCK_RECEPTOR_SDA1 = [0.951717212085, 0.619741731921]
MOCK_RECEPTOR_SDE1 = [0.460607106742, 0.85383818835]

MOCK_RECEPTOR_MEAN2 = [2.06854058709, 1.06221881356]
MOCK_RECEPTOR_SDA2 = [1.151717212085, 0.519741731921]
MOCK_RECEPTOR_SDE2 = [0.760607106742, 0.65383818835]

# HEAT MAP
PIXEL_PER_Q_UNIT = 20
