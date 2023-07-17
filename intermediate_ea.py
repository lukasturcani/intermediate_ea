import argparse
import itertools
import logging
import pathlib
import typing
from collections.abc import Iterable, Iterator
from functools import partial
from typing import Any, cast

import atomlite
import numpy as np
import numpy.typing as npt
import rdkit.Chem.AllChem as rdkit
import stk
from rdkit import RDLogger
from rdkit.Chem.GraphDescriptors import BertzCT

rdkit_logger = RDLogger.logger()
rdkit_logger.setLevel(RDLogger.CRITICAL)
logger = logging.getLogger(__name__)


T = typing.TypeVar("T", bound=stk.MoleculeRecord)

MoleculeRecord: typing.TypeAlias = stk.MoleculeRecord[stk.polymer.Linear]


def get_building_blocks(
    path: pathlib.Path,
    functional_group_factory: stk.FunctionalGroupFactory,
    generator: np.random.Generator,
) -> Iterator[stk.BuildingBlock]:
    with open(path, "r") as f:
        content = f.readlines()

    for smiles in content:
        molecule = rdkit.AddHs(rdkit.MolFromSmiles(smiles))
        molecule.AddConformer(rdkit.Conformer(molecule.GetNumAtoms()))
        rdkit.Kekulize(molecule)
        building_block = stk.BuildingBlock.init_from_rdkit_mol(
            molecule=molecule,
            functional_groups=functional_group_factory,
        )
        yield building_block.with_position_matrix(
            position_matrix=get_position_matrix(building_block, generator),
        )


def get_position_matrix(
    molecule: stk.BuildingBlock,
    generator: np.random.Generator,
) -> npt.NDArray[np.float64]:
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


def get_initial_population(
    fluoros: Iterable[stk.BuildingBlock],
    bromos: Iterable[stk.BuildingBlock],
) -> Iterator[MoleculeRecord]:
    for fluoro, bromo in itertools.product(fluoros, bromos):
        yield stk.MoleculeRecord(
            topology_graph=stk.polymer.Linear(
                building_blocks=(fluoro, bromo),
                repeating_unit="AB",
                num_repeating_units=1,
            ),
        )


def get_num_rotatable_bonds(
    database: atomlite.Database,
    record: MoleculeRecord,
) -> int:
    key = get_key(record)
    path = "$.ea.num_rotatable_bonds"
    num_rotatable_bonds = cast(int | None, database.get_property(key, path))
    if num_rotatable_bonds is None:
        rdkit_molecule = record.get_molecule().to_rdkit_mol()
        rdkit.SanitizeMol(rdkit_molecule)
        num_rotatable_bonds = rdkit.CalcNumRotatableBonds(rdkit_molecule)
        database.set_property(key, path, num_rotatable_bonds)
    return num_rotatable_bonds


def get_complexity(
    database: atomlite.Database,
    record: MoleculeRecord,
) -> float:
    key = get_key(record)
    path = "$.ea.complexity"
    complexity = cast(float | None, database.get_property(key, path))
    if complexity is None:
        rdkit_molecule = record.get_molecule().to_rdkit_mol()
        rdkit.SanitizeMol(rdkit_molecule)
        complexity = BertzCT(rdkit_molecule)
        database.set_property(key, path, complexity)
    return complexity


def get_num_bad_rings(
    database: atomlite.Database,
    record: MoleculeRecord,
) -> int:
    key = get_key(record)
    path = "$.ea.num_bad_rings"
    num_bad_rings = cast(int | None, database.get_property(key, path))
    if num_bad_rings is None:
        rdkit_molecule = record.get_molecule().to_rdkit_mol()
        rdkit.SanitizeMol(rdkit_molecule)
        num_bad_rings = sum(
            1 for ring in rdkit.GetSymmSSSR(rdkit_molecule) if len(ring) < 5
        )
        database.set_property(key, path, num_bad_rings)
    return num_bad_rings


def get_functional_group_type(building_block: stk.BuildingBlock) -> type:
    (functional_group,) = building_block.get_functional_groups(0)
    return functional_group.__class__


def is_fluoro(building_block: stk.BuildingBlock) -> bool:
    (functional_group,) = building_block.get_functional_groups(0)
    return functional_group.__class__ is stk.Fluoro


def is_bromo(building_block: stk.BuildingBlock) -> bool:
    (functional_group,) = building_block.get_functional_groups(0)
    return functional_group.__class__ is stk.Bromo


def write(
    molecule: stk.ConstructedMolecule,
    path: pathlib.Path,
    generator: np.random.Generator,
) -> None:
    rdkit_molecule = molecule.to_rdkit_mol()
    rdkit.SanitizeMol(rdkit_molecule)
    rdkit_molecule = rdkit.RemoveHs(rdkit_molecule)
    building_block = stk.BuildingBlock.init_from_rdkit_mol(
        molecule=rdkit_molecule,
    )
    building_block.with_position_matrix(
        position_matrix=get_position_matrix(building_block, generator),
    ).write(path)


def get_key(record: MoleculeRecord) -> str:
    return stk.Inchi().get_key(record.get_molecule())


def get_entry(record: MoleculeRecord) -> atomlite.Entry:
    return atomlite.Entry.from_rdkit(
        key=get_key(record),
        molecule=record.get_molecule().to_rdkit_mol(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database",
        help="Path to an AtomLite database holding EA results.",
        type=pathlib.Path,
        default=pathlib.Path("intermediate_ea.db"),
    )
    parser.add_argument(
        "--fluoros",
        help="Path to file holding SMILES of Fluoro building blocks.",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent / "fluoros.txt",
    )
    parser.add_argument(
        "--bromos",
        help="Path to file holding SMILES of Bromo building blocks.",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent / "bromos.txt",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    generator = np.random.default_rng(4)

    logger.info("Making building blocks.")

    # Load the building block databases.
    fluoros = tuple(
        get_building_blocks(
            path=args.fluoros,
            functional_group_factory=stk.FluoroFactory(),
            generator=generator,
        )
    )
    bromos = tuple(
        get_building_blocks(
            path=args.bromos,
            functional_group_factory=stk.BromoFactory(),
            generator=generator,
        )
    )

    initial_population = tuple(get_initial_population(fluoros[:5], bromos[:5]))

    # Write the initial population.
    initial_population_directory = pathlib.Path("initial_population")
    initial_population_directory.mkdir(exist_ok=True, parents=True)
    for i, record in enumerate(initial_population):
        write(
            molecule=record.get_molecule(),
            path=initial_population_directory / f"initial_{i}.mol",
            generator=generator,
        )

    db = atomlite.Database(args.database)

    # Plot selections.
    generation_selector: stk.Best[MoleculeRecord] = stk.Best(
        num_batches=25,
        duplicate_molecules=False,
    )
    stk.SelectionPlotter("generation_selection", generation_selector)

    mutation_selector: stk.Roulette[MoleculeRecord] = stk.Roulette(
        num_batches=5,
        random_seed=generator,
    )
    stk.SelectionPlotter("mutation_selection", mutation_selector)

    crossover_selector: stk.Roulette[MoleculeRecord] = stk.Roulette(
        num_batches=3,
        batch_size=2,
        random_seed=generator,
    )
    stk.SelectionPlotter("crossover_selection", crossover_selector)

    fitness_calculator = stk.PropertyVector(
        property_functions=(
            partial(get_num_rotatable_bonds, db),
            partial(get_complexity, db),
            partial(get_num_bad_rings, db),
        ),
    )

    fitness_normalizer: stk.NormalizerSequence[MoleculeRecord] = stk.NormalizerSequence(
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
                    random_seed=generator,
                ),
                stk.SimilarBuildingBlock(
                    building_blocks=fluoros,
                    is_replaceable=is_fluoro,
                    random_seed=generator,
                ),
                stk.RandomBuildingBlock(
                    building_blocks=bromos,
                    is_replaceable=is_bromo,
                    random_seed=generator,
                ),
                stk.SimilarBuildingBlock(
                    building_blocks=bromos,
                    is_replaceable=is_bromo,
                    random_seed=generator,
                ),
            ),
            random_seed=generator,
        ),
        crosser=stk.GeneticRecombination(
            get_gene=get_functional_group_type,
        ),
        generation_selector=generation_selector,
        mutation_selector=mutation_selector,
        crossover_selector=crossover_selector,
        fitness_normalizer=fitness_normalizer,
    )

    logger.info("Starting EA.")

    generations = []
    num_rotatable_bonds = []
    fitness_values: dict[MoleculeRecord, Any] = {}
    for generation in ea.get_generations(50):
        generations.append(list(generation.get_molecule_records()))
        db.update_entries(map(get_entry, generation.get_molecule_records()))
        num_rotatable_bonds.append(
            [
                cast(int, db.get_property(get_key(record), "$.ea.num_rotatable_bonds"))
                for record in generation.get_molecule_records()
            ]
        )
        fitness_values.update(
            (record, fitness_value.raw)
            for record, fitness_value in generation.get_fitness_values().items()
        )

    # Write the final population.
    final_population_directory = pathlib.Path("final_population")
    final_population_directory.mkdir(exist_ok=True, parents=True)
    for i, record in enumerate(generation.get_molecule_records()):
        write(
            molecule=record.get_molecule(),
            path=final_population_directory / f"final_{i}.mol",
            generator=generator,
        )

    logger.info("Making fitness plot.")

    # Normalize the fitness values across the entire EA before
    # plotting the fitness values.
    normalized_fitness_values = fitness_normalizer.normalize(fitness_values)
    fitness_values_by_generation = [
        [normalized_fitness_values[record] for record in generation]
        for generation in generations
    ]

    fitness_progress = stk.ProgressPlotter(
        property=fitness_values_by_generation,
        y_label="Fitness Value",
    )
    fitness_progress.write("fitness_progress.png")
    fitness_progress.get_plot_data().to_csv("fitness_progress.csv")

    logger.info("Making rotatable bonds plot.")

    rotatable_bonds_progress = stk.ProgressPlotter(
        property=num_rotatable_bonds,
        y_label="Number of Rotatable Bonds",
    )
    rotatable_bonds_progress.write("rotatable_bonds_progress.png")


if __name__ == "__main__":
    main()
