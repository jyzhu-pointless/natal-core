"""Numerical oracle for the low-level WF compatibility fallback."""

import numpy as np
import pytest

import natal as nt
from natal.frontend.model import build_discrete_engine_config


@pytest.mark.parametrize("compatibility", [1.0, 0.0])
def test_wf_unconstrained_sex_branch_conserves_offspring(compatibility: float) -> None:
    """An unconstrained ztype divides 100 offspring equally even for row sums 1+1 or 0+0.

    Deliberately exercise the engine's fallback contract through an internal
    configuration: real XY masks constrain every valid genotype and therefore
    do not reach this branch.  The offspring tensor remains normalized and
    unchanged; compatibility values select the allocation ratio only.
    """
    species = nt.Species.from_dict(
        name="independent_wf_fallback", structure={"c": {"L": ["A"]}},
    )
    config = build_discrete_engine_config(
        n_genotypes=1, n_gtypes=1, n_glabs=1,
        stochastic=False, continuous_sampling=False,
        zygotes_to_gametes_map=np.ones((2, 1, 1)),
        gametes_to_zygotes_map=np.ones((1, 1, 1)),
        eggs_per_female=1, fixed_egg_count=True,
        juvenile_growth_mode=nt.NO_COMPETITION,
        sex_ratio=0.9,
    )._replace(
        extreme_speed_mode=3,
        has_sex_chromosomes=True,
        female_only_by_sex_chrom=np.zeros(1, dtype=np.bool_),
        male_only_by_sex_chrom=np.zeros(1, dtype=np.bool_),
        female_ztype_compatibility=np.full(1, compatibility),
        male_ztype_compatibility=np.full(1, compatibility),
    )
    pop = nt.DiscreteGenerationPopulation(
        species=species, population_config=config,
        initial_individual_count={"female": {"A|A": 100}, "male": {"A|A": 100}},
    )
    pop.run(1)
    np.testing.assert_array_equal(pop.state.individual_count.sum(axis=(1, 2)), [50.0, 50.0])
