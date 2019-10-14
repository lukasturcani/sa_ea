import stk
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        metavar='population_path',
        help='Path to an stk Population dump file.',
        nargs='+',
        dest='population_paths',
    )

    args = parser.parse_args()
    pops = []
    for path in args.population_paths:
        pops.append(set(stk.Population.load(path, True)))

    for i, pop in enumerate(args.population_paths):
        print(i, pop)

    overlaps = [
        [len(pop1 & pop2) for pop2 in pops]
        for pop1 in pops
    ]
    df = pd.DataFrame(overlaps)
    print(df)


if __name__ == '__main__':
    main()
