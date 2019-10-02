import stk
import pywindow as pywindow
from glob import iglob
from os.path import join
import logging
import numpy as np
import pickle
import rdkit.Chem.AllChem as rdkit


random_seed = 13
xtb_path = '/setup_environment/xtb_190418/bin/xtb'
num_processes = 25

# #####################################################################
# Set logging level.
# #####################################################################

logging_level = logging.DEBUG

# #####################################################################
# Toggle the writing of a log file.
# #####################################################################

log_file = True

# #####################################################################
# Toggle the dumping of generated molecules.
# #####################################################################

database_dump = True

# #####################################################################
# Toggle the dumping of EA generations.
# #####################################################################

progress_dump = True

# #####################################################################
# Toggle the dumping of molecules at every generation.
# #####################################################################

debug_dumps = False

# #####################################################################
# Make a tar archive of the output.
# #####################################################################

tar_output = True

# #####################################################################
# Initial population.
# #####################################################################

db = '/setup_environment'

amines = [
    stk.BuildingBlock.init_from_file(path, ['amine'])
    for path in iglob(join(db, 'amines2f', '*.mol'))
]

aldehydes = [
    stk.BuildingBlock.init_from_file(path, ['aldehyde'])
    for path in iglob(join(db, 'aldehydes3f', '*.mol'))
]

# Create the initial population.
population_size = 25
population = stk.EAPopulation.init_random(
    building_blocks=[amines, aldehydes],
    topology_graphs=[stk.cage.FourPlusSix()],
    size=population_size,
    random_seed=random_seed,
)

# #####################################################################
# Selector for selecting the next generation.
# #####################################################################

generation_selector = stk.SelectorSequence(
    selector1=stk.AboveAverage(duplicate_mols=False),
    selector2=stk.RemoveMolecules(
        remover=stk.AboveAverage(duplicate_mols=False),
        selector=stk.Roulette(
            duplicate_mols=False,
            random_seed=random_seed,
        ),
    ),
    num_batches=population_size,
)

# #####################################################################
# Selector for selecting parents.
# #####################################################################

above_average = stk.AboveAverage(
    batch_size=2,
    duplicate_batches=False,
)
crossover_selector = stk.SelectorSequence(
    selector1=above_average,
    selector2=stk.RemoveBatches(
        remover=above_average,
        selector=stk.StochasticUniversalSampling(
            num_batches=5,
            batch_size=2,
            duplicate_batches=False,
            random_seed=random_seed,
        ),
    ),
)

# #####################################################################
# Selector for selecting molecules for mutation.
# #####################################################################

mutation_selector = stk.Roulette(
    num_batches=10,
    random_seed=random_seed,
)


# #####################################################################
# Crosser.
# #####################################################################

crosser = stk.GeneticRecombination(
    key=lambda mol: mol.func_groups[0].fg_type.name,
    random_seed=random_seed,
    use_cache=True,
)

# #####################################################################
# Mutator.
# #####################################################################

mutator = stk.RandomMutation(
    stk.RandomBuildingBlock(
        building_blocks=amines,
        key=lambda mol: mol.func_groups[0].fg_type.name == 'amine',
        duplicate_building_blocks=False,
        random_seed=random_seed,
        use_cache=True,
    ),
    stk.SimilarBuildingBlock(
        building_blocks=amines,
        key=lambda mol: mol.func_groups[0].fg_type.name == 'amine',
        duplicate_building_blocks=False,
        random_seed=random_seed,
        use_cache=True,
    ),
    stk.RandomBuildingBlock(
        building_blocks=aldehydes,
        key=lambda mol: mol.func_groups[0].fg_type.name == 'aldehyde',
        duplicate_building_blocks=False,
        random_seed=random_seed,
        use_cache=True,
    ),
    stk.SimilarBuildingBlock(
        building_blocks=aldehydes,
        key=lambda mol: mol.func_groups[0].fg_type.name == 'aldehyde',
        duplicate_building_blocks=False,
        random_seed=random_seed,
        use_cache=True,
    ),
    random_seed=random_seed,
)

# #####################################################################
# Optimizer.
# #####################################################################

optimizer = stk.OptimizerSequence(
    stk.MacroModelForceField(
        macromodel_path='/opt/schrodinger2018-1',
        restricted=True,
        use_cache=True,
    ),
    stk.MacroModelForceField(
        macromodel_path='/opt/schrodinger2018-1',
        restricted=False,
        use_cache=True,
    ),
    stk.XTB(
        xtb_path=xtb_path,
        calculate_hessian=False,
        unlimited_memory=False,
        use_cache=True,
    ),
    use_cache=True,
)

# #####################################################################
# Fitness Calculator.
# #####################################################################


def pore_diameter(mol):
    pw_mol = pywindow.Molecule.load_rdkit_mol(mol.to_rdkit_mol())
    return pw_mol.calculate_pore_diameter()


def window_std(mol):
    pw_mol = pywindow.Molecule.load_rdkit_mol(mol.to_rdkit_mol())
    windows = pw_mol.calculate_windows()
    if windows is not None and len(windows) > 3:
        return np.std(windows)


fitness_calculator = stk.PropertyVector(
    pore_diameter,
    window_std,
)

fitness_normalizer = stk.NormalizerSequence(
    # This coefficient needs to make sense for conformer_analysis.
    stk.Power([1, -1]),
    stk.ScaleByMean(),
    stk.Multiply([1.0, 1.0]),
    stk.Sum(),
)


# #####################################################################
# Exit condition.
# #####################################################################

exiter = stk.NumGenerations(60)

# #####################################################################
# Make plotters.
# #####################################################################

plotters = [
    stk.ProgressPlotter(
        filename='fitness_plot',
        property_fn=lambda mol: mol.fitness,
        y_label='Fitness',
    ),
    stk.ProgressPlotter(
        filename='max_window_diameter',
        property_fn=lambda mol: mol.max_window_diameter,
        y_label='Maximum Window Diameter / A',
    ),
    stk.ProgressPlotter(
        filename='pore_diameter',
        property_fn=lambda mol: mol.pore_diameter,
        y_label='Pore Diameter / A',
    ),
    stk.ProgressPlotter(
        filename='sa_score',
        property_fn=lambda mol: mol.sa_score,
        y_label='SA Score / arb. unit',
    )
]

stk.SelectionPlotter(
    filename='generational_selection',
    selector=generation_selector,
)
stk.SelectionPlotter(
    filename='crossover_selection',
    selector=crossover_selector,
)
stk.SelectionPlotter(
    filename='mutation_selection',
    selector=mutation_selector,
)