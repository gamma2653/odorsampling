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


# odor concentration (driving the size of the odors)
ODOR_CONCENTRATION = 1e-5
# ODOR_CONCENTRATION = 1e-8

# location distributions control params, eg., uniform, gaussian...1 and only 1 type needs to be true at any given time
DIST_TYPE_GAUSS = False
MU = 6
SIG = 12

DIST_TYPE_UNIF = True


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

