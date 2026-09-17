"""P9: Wolbachia maternal inheritance and carrier fitness scaling.

Claim: the Wolbachia preset tags maternal gametes of infected females with
the cytoplasmic label, so every offspring of an infected mother carries the
infected somatic slab regardless of the father's infection state, while
offspring of uninfected mothers stay uninfected.  The carrier fitness
patch (viability_scaling) scales only infected carriers.

Reference: the documented maternal-transmission contract (maternal gamete
relabeling) and patch scoping; expectations computed from egg counts.

Wrong results rejected: paternal transmission of the infection, infection
label lost across a generation, fitness scaling applied to uninfected
individuals or to both slabs.
"""

from __future__ import annotations

import natal as nt
import numpy as np
from natal.frontend.presets.cytoplasmic import Wolbachia

TOL = 1e-9


def _species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT"]}},
        gamete_labels=["default", "wolbachia"],
        somatic_labels=["normal", "infected"],
    )


def _slab_totals(pop):
    """Return (normal_total_both_sexes, infected_total_both_sexes) adults."""
    normal = infected = 0.0
    for g, slab in pop.registry.index_to_ztype:
        idx = pop.registry.ztype_index(g, slab)
        total = float(np.asarray(pop.state.individual_count[:, 1, idx]).sum())
        if slab == "infected":
            infected += total
        else:
            normal += total
    return normal, infected


def test_maternal_transmission_and_viability_scaling() -> None:
    pop = (
        nt.DiscreteGenerationPopulation.setup(
            species=_species("QC0915_p09a"), name="QC0915_p09a", stochastic=False
        )
        .presets(Wolbachia(name="QC0915_wol_a", viability_scaling=0.9))
        .initial_state(
            individual_count={
                "female": {"WT|WT@infected": 500, "WT|WT": 500},
                "male": {"WT|WT@infected": 500, "WT|WT": 500},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2.0)
        .competition(juvenile_growth_mode="no_competition")
        .build()
    )
    pop.run(1)
    normal, infected = _slab_totals(pop)
    # 500 infected mothers -> 1000 infected newborns -> 500 per sex -> 450
    # after 0.9 viability.  500 normal mothers -> 500 normal adults; the
    # fathers' infection state is irrelevant.
    assert abs(infected - 900.0) < TOL
    assert abs(normal - 1000.0) < TOL


def test_infected_fathers_cannot_transmit() -> None:
    pop = (
        nt.DiscreteGenerationPopulation.setup(
            species=_species("QC0915_p09b"), name="QC0915_p09b", stochastic=False
        )
        .presets(Wolbachia(name="QC0915_wol_b", viability_scaling=1.0))
        .initial_state(
            individual_count={
                "female": {"WT|WT": 500},
                "male": {"WT|WT@infected": 500},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2.0)
        .competition(juvenile_growth_mode="no_competition")
        .build()
    )
    pop.run(1)
    normal, infected = _slab_totals(pop)
    assert abs(normal - 1000.0) < TOL
    assert infected == 0.0
