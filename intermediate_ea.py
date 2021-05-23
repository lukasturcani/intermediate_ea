import argparse
import stk
import rdkit.Chem.AllChem as rdkit
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit import RDLogger
import pymongo
import numpy as np
import itertools as it
import logging
import pathlib

rdkit_logger = RDLogger.logger()
rdkit_logger.setLevel(RDLogger.CRITICAL)
logger = logging.getLogger(__name__)


def get_building_blocks(path, functional_group_factory):
    with open(path, 'r') as f:
        content = f.readlines()

    for smiles in content:
        molecule = rdkit.AddHs(rdkit.MolFromSmiles(smiles))
        molecule.AddConformer(rdkit.Conformer(molecule.GetNumAtoms()))
        rdkit.Kekulize(molecule)
        building_block = stk.BuildingBlock.init_from_rdkit_mol(
            molecule=molecule,
            functional_groups=[functional_group_factory],
        )
        yield building_block.with_position_matrix(
            position_matrix=get_position_matrix(building_block),
        )


def get_position_matrix(molecule):
    generator = np.random.RandomState(4)
    position_matrix = generator.uniform(
        low=-500,
        high=500,
        size=(molecule.get_num_atoms(), 3),
    )
    molecule = molecule.with_position_matrix(position_matrix)
    rdkit_molecule = molecule.to_rdkit_mol()
    rdkit.SanitizeMol(rdkit_molecule)
    rdkit.Compute2DCoords(rdkit_molecule)
    try:
        rdkit.MMFFOptimizeMolecule(rdkit_molecule)
    except Exception:
        pass
    return rdkit_molecule.GetConformer().GetPositions()


def get_initial_population(fluoros, bromos):
    for fluoro, bromo in it.product(fluoros[:5], bromos[:5]):
        yield stk.MoleculeRecord(
            topology_graph=stk.polymer.Linear(
                building_blocks=(fluoro, bromo),
                repeating_unit='AB',
                num_repeating_units=1,
            ),
        )


def get_num_rotatable_bonds(molecule):
    rdkit_molecule = molecule.to_rdkit_mol()
    rdkit.SanitizeMol(rdkit_molecule)
    return rdkit.CalcNumRotatableBonds(rdkit_molecule)


def get_complexity(molecule):
    rdkit_molecule = molecule.to_rdkit_mol()
    rdkit.SanitizeMol(rdkit_molecule)
    return BertzCT(rdkit_molecule)


def get_num_bad_rings(molecule):
    rdkit_molecule = molecule.to_rdkit_mol()
    rdkit.SanitizeMol(rdkit_molecule)
    return sum(
        1
        for ring in rdkit.GetSymmSSSR(rdkit_molecule) if len(ring) < 5
    )


def get_functional_group_type(building_block):
    functional_group, = building_block.get_functional_groups(0)
    return functional_group.__class__


def is_fluoro(building_block):
    functional_group, = building_block.get_functional_groups(0)
    return functional_group.__class__ is stk.Fluoro


def is_bromo(building_block):
    functional_group, = building_block.get_functional_groups(0)
    return functional_group.__class__ is stk.Bromo


def normalize_generations(
    fitness_calculator,
    fitness_normalizer,
    generations,
):
    population = tuple(
        record.with_fitness_value(
            fitness_value=fitness_calculator.get_fitness_value(
                molecule=record.get_molecule(),
            ),
            normalized=False,
        )
        for generation in generations
        for record in generation.get_molecule_records()
    )
    population = tuple(fitness_normalizer.normalize(population))

    num_generations = len(generations)
    population_size = sum(
        1 for _ in generations[0].get_molecule_records()
    )
    num_molecules = num_generations*population_size

    for generation, start in zip(
        generations,
        range(0, num_molecules, population_size),
    ):
        end = start + population_size
        yield stk.Generation(
            molecule_records=population[start:end],
            mutation_records=tuple(generation.get_mutation_records()),
            crossover_records=tuple(
                generation.get_crossover_records()
            ),
        )


def write(molecule, path):
    rdkit_molecule = molecule.to_rdkit_mol()
    rdkit.SanitizeMol(rdkit_molecule)
    rdkit_molecule = rdkit.RemoveHs(rdkit_molecule)
    building_block = stk.BuildingBlock.init_from_rdkit_mol(
        molecule=rdkit_molecule,
    )
    building_block.with_position_matrix(
        position_matrix=get_position_matrix(building_block),
    ).write(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mongodb_uri',
        help='The MongoDB URI for the database to connect to.',
        default='mongodb://localhost:27017/'
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Use a random seed to get reproducible results.
    random_seed = 4
    generator = np.random.RandomState(random_seed)

    logger.info('Making building blocks.')

    # Load the building block databases.
    fluoros = tuple(get_building_blocks(
        path=pathlib.Path(__file__).parent / 'fluoros.txt',
        functional_group_factory=stk.FluoroFactory(),
    ))
    bromos = tuple(get_building_blocks(
        path=pathlib.Path(__file__).parent / 'bromos.txt',
        functional_group_factory=stk.BromoFactory(),
    ))

    initial_population = tuple(get_initial_population(fluoros, bromos))
    # Write the initial population.
    for i, record in enumerate(initial_population):
        write(record.get_molecule(), f'initial_{i}.mol')

    client = pymongo.MongoClient(args.mongodb_uri)
    db = stk.ConstructedMoleculeMongoDb(client)
    fitness_db = stk.ValueMongoDb(client, 'fitness_values')

    # Plot selections.
    generation_selector = stk.Best(
        num_batches=25,
        duplicate_molecules=False,
    )
    stk.SelectionPlotter('generation_selection', generation_selector)

    mutation_selector = stk.Roulette(
        num_batches=5,
        random_seed=generator.randint(0, 1000),
    )
    stk.SelectionPlotter('mutation_selection', mutation_selector)

    crossover_selector = stk.Roulette(
        num_batches=3,
        batch_size=2,
        random_seed=generator.randint(0, 1000),
    )
    stk.SelectionPlotter('crossover_selection', crossover_selector)

    fitness_calculator = stk.PropertyVector(
        property_functions=(
            get_num_rotatable_bonds,
            get_complexity,
            get_num_bad_rings,
        ),
        input_database=fitness_db,
        output_database=fitness_db,
    )

    fitness_normalizer = stk.NormalizerSequence(
        fitness_normalizers=(
            # Prevent division by 0 error in DivideByMean, by ensuring
            # a value of each property to be at least 1.
            stk.Add((1, 1, 1)),
            stk.DivideByMean(),
            # Obviously, because all coefficients are equal, the
            # Multiply normalizer does not need to be here. However,
            # it's here to show that you can easily change the relative
            # importance of each component of the fitness value, by
            # changing the values of the coefficients.
            stk.Multiply((1, 1, 1)),
            stk.Sum(),
            stk.Power(-1),
        ),
    )

    ea = stk.EvolutionaryAlgorithm(
        num_processes=1,
        initial_population=initial_population,
        fitness_calculator=fitness_calculator,
        mutator=stk.RandomMutator(
            mutators=(
                stk.RandomBuildingBlock(
                    building_blocks=fluoros,
                    is_replaceable=is_fluoro,
                    random_seed=generator.randint(0, 1000),
                ),
                stk.SimilarBuildingBlock(
                    building_blocks=fluoros,
                    is_replaceable=is_fluoro,
                    random_seed=generator.randint(0, 1000),
                ),
                stk.RandomBuildingBlock(
                    building_blocks=bromos,
                    is_replaceable=is_bromo,
                    random_seed=generator.randint(0, 1000),
                ),
                stk.SimilarBuildingBlock(
                    building_blocks=bromos,
                    is_replaceable=is_bromo,
                    random_seed=generator.randint(0, 1000),
                ),
            ),
            random_seed=generator.randint(0, 1000),
        ),
        crosser=stk.GeneticRecombination(
            get_gene=get_functional_group_type,
        ),
        generation_selector=generation_selector,
        mutation_selector=mutation_selector,
        crossover_selector=crossover_selector,
        fitness_normalizer=fitness_normalizer,
    )

    logger.info('Starting EA.')

    generations = []
    for generation in ea.get_generations(50):
        for record in generation.get_molecule_records():
            db.put(record.get_molecule())
        generations.append(generation)

    # Write the final population.
    for i, record in enumerate(generation.get_molecule_records()):
        write(record.get_molecule(), f'final_{i}.mol')

    logger.info('Making fitness plot.')

    # Normalize the fitness values across the entire EA before
    # plotting the fitness values.
    generations = tuple(normalize_generations(
        fitness_calculator=fitness_calculator,
        fitness_normalizer=fitness_normalizer,
        generations=generations,
    ))

    fitness_progress = stk.ProgressPlotter(
        generations=generations,
        get_property=lambda record: record.get_fitness_value(),
        y_label='Fitness Value',
    )
    fitness_progress.write('fitness_progress.png')

    logger.info('Making rotatable bonds plot.')

    rotatable_bonds_progress = stk.ProgressPlotter(
        generations=generations,
        get_property=lambda record:
            get_num_rotatable_bonds(record.get_molecule()),
        y_label='Number of Rotatable Bonds',
    )
    rotatable_bonds_progress.write('rotatable_bonds_progress.png')


if __name__ == '__main__':
    main()
