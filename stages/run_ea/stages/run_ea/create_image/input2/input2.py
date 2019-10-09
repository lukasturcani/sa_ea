import stk
import pywindow as pywindow
from glob import iglob
from os.path import join
import logging
import numpy as np
import pickle
import rdkit.Chem.AllChem as rdkit


random_seed = 13
xtb_path = '/home/lt912/xtb_190418/bin/xtb'
num_processes = 20

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

db = (
    '/home/lt912/sa_ea/stages/run_ea/stages/setup_environment'
    '/create_image/setup_stage/setup_environment'
)

amines = [
    stk.BuildingBlock.init_from_file(path, ['amine'], True)
    for path in iglob(join(db, 'amines2f', '*.mol'))
]

aldehydes = [
    stk.BuildingBlock.init_from_file(path, ['aldehyde'], True)
    for path in iglob(join(db, 'aldehydes3f', '*.mol'))
]

# Create the initial population.
population_size = 25
population = stk.EAPopulation.init_random(
    building_blocks=[amines, aldehydes],
    topology_graphs=[stk.cage.FourPlusSix()],
    size=population_size,
    random_seed=random_seed,
    use_cache=True,
)

# #####################################################################
# Selector for selecting the next generation.
# #####################################################################

generation_selector = stk.Sequence(
    stk.AboveAverage(duplicate_mols=False),
    stk.RemoveMolecules(
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
crossover_selector = stk.Sequence(
    above_average,
    stk.RemoveBatches(
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

mutator = stk.Random(
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

optimizer = stk.Sequence(
    stk.MacroModelForceField(
        macromodel_path='/home/lt912/schrodinger2017-4',
        restricted=True,
        use_cache=True,
    ),
    stk.MacroModelForceField(
        macromodel_path='/home/lt912/schrodinger2017-4',
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
    mol.pore_diameter = pw_mol.calculate_pore_diameter()
    return mol.pore_diameter


def window_std(mol):
    pw_mol = pywindow.Molecule.load_rdkit_mol(mol.to_rdkit_mol())
    windows = pw_mol.calculate_windows()
    mol.window_std = None
    if windows is not None and len(windows) > 3:
        mol.window_std = np.std(windows)
    return mol.window_std


def fingerprint(mol):
    rdkit_mol = mol.to_rdkit_mol()
    rdkit.SanitizeMol(rdkit_mol)
    info = {}
    fp = rdkit.GetMorganFingerprintAsBitVect(
        mol=rdkit_mol,
        radius=8,
        nBits=512,
        bitInfo=info,
    )
    fp = list(fp)
    for bit, activators in info.items():
        fp[bit] = len(activators)
    return [fp]


with open('/home/lt912/sa_model.pkl', 'rb') as f:
    clf = pickle.load(f)


def sa_score(mol):
    mol.sa_score = sum(
        clf.predict(fingerprint(bb))[0]
        for bb in mol.get_building_blocks()
    ) + 1
    return mol.sa_score


fitness_calculator = stk.PropertyVector(
    pore_diameter,
    window_std,
    sa_score,
)


def valid_fitness(population, mol):
    return None not in population.get_fitness_values()[mol]


fitness_normalizer = stk.Sequence(
    stk.Power([1, -1, 1], filter=valid_fitness),
    stk.DivideByMean(filter=valid_fitness),
    stk.Multiply([1.0, 1.0, 10.0], filter=valid_fitness),
    stk.Sum(filter=valid_fitness),
    stk.ReplaceFitness(
        replacement_fn=lambda population:
            min(
                f for _, f in population.get_fitness_values().items()
                if not isinstance(f, list)
            ) / 2,
        filter=lambda p, m:
            isinstance(p.get_fitness_values()[m], list),
    )
)


# #####################################################################
# Exit condition.
# #####################################################################

terminator = stk.NumGenerations(60)

# #####################################################################
# Make plotters.
# #####################################################################

plotters = [
    stk.ProgressPlotter(
        filename='fitness_plot',
        property_fn=lambda progress, mol:
            progress.get_fitness_values()[mol],
        y_label='Fitness',
        filter=lambda progress, mol:
            progress.get_fitness_values()[mol],
        progress_fn=lambda progress:
            progress.set_fitness_values_from_calculators(
                fitness_calculator=fitness_calculator,
                fitness_normalizer=fitness_normalizer,
                num_processes=num_processes,
            )
    ),
    stk.ProgressPlotter(
        filename='window_std',
        property_fn=lambda mol: mol.window_std,
        y_label='Std. Dev. of Window Diameters / A',
        filter=lambda progress, mol:
            mol.window_std is not None,
    ),
    stk.ProgressPlotter(
        filename='pore_diameter',
        property_fn=lambda progress, mol: mol.pore_diameter,
        y_label='Pore Diameter / A',
    ),
    stk.ProgressPlotter(
        filename='sa_score',
        property_fn=lambda progress, mol: mol.sa_score,
        y_label='SA Score / arb. unit',
    ),
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
