import argparse
import stk
import pickle
import rdkit.Chem.AllChem as rdkit
from os.path import join


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


def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    def inner(mol):
        return sum(
            model.predict_proba(fingerprint(bb))[0][1]
            for bb in mol.get_building_blocks()
        ) + 1

    return inner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'population_path',
        help='Path to an stk Population dump file.',
    )
    parser.add_argument(
        'sa_model_path',
        help='Path to a pickled estimator predicting SA.',
    )
    parser.add_argument(
        'output_directory',
        help='Path to the directory into which output is written.'
    )

    args = parser.parse_args()
    population = stk.Population.load(args.population_path, True)
    sa = load_model(args.sa_model_path)

    sorted_population = sorted(population, key=sa, reverse=True)
    mean_sa = sum(map(sa, population)) / len(population)

    max_sa = sa(sorted_population[0])
    print(args.population_path)
    print(f'Maximum SA in the population was {max_sa}.')
    print(f'Mean SA in the population was {mean_sa}.')
    for rank, mol in enumerate(sorted_population):
        mol.write(join(args.output_directory, f'sa_rank_{rank}.mol'))


if __name__ == '__main__':
    main()
