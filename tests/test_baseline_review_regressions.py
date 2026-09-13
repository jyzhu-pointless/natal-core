"""Independent CR-7/CR-9 review counterexamples for baseline contracts."""

import numpy as np
import pytest

import natal as nt


def test_baseline_rejects_out_of_range_rate_written_through_view() -> None:
    """Shared-view writes must receive the setter's [0, 0.5] validation."""
    species = nt.Species.from_dict(
        name="review_cr7_rate_bound",
        structure={"c": {"l1": ["A", "a"], "l2": ["B", "b"]}},
    )
    species.get_config_blueprint()
    np.asarray(species.chromosomes[0].recombination_map)[:] = 0.9
    with pytest.raises(ValueError):
        species.get_config_blueprint()


def test_returned_baseline_cannot_poison_future_compilation() -> None:
    """A caller's mutation must not replace the internal Mendelian baseline."""
    species = nt.Species.from_dict(
        name="review_cr7_ownership", structure={"c": {"l": ["A", "a"]}}
    )
    first = species.get_config_blueprint()
    expected = first["zygotes_to_gametes_map"].copy()
    try:
        first["zygotes_to_gametes_map"].fill(0.0)
    except ValueError:
        pass  # A truly immutable public result is also a valid solution.
    np.testing.assert_array_equal(
        species.get_config_blueprint()["zygotes_to_gametes_map"], expected
    )


def test_baseline_rejects_recombination_locus_mapping_mismatch() -> None:
    """The dependency includes which adjacent loci each rate belongs to."""
    species = nt.Species.from_dict(
        name="review_cr7_locus_mapping",
        structure={
            "c": {"l1": ["A", "a"], "l2": ["B", "b"], "l3": ["C", "c"]}
        },
    )
    recombination_map = species.chromosomes[0].recombination_map
    recombination_map[:] = [0.1, 0.3]
    species.get_config_blueprint()
    recombination_map.loci_names[:] = ["missing", "l2", "l3"]
    with pytest.raises(ValueError):
        species.get_config_blueprint()


@pytest.mark.parametrize(
    "entry",
    ["iter_genotypes", "iter_haploid_genotypes", "get_maternal_haploid_genotypes"],
)
def test_empty_chromosome_rejected_by_all_enumeration_entries(entry: str) -> None:
    """Enumeration must reject an incomplete chromosome rather than return []."""
    species = nt.Species.from_dict(
        name=f"review_cr9_{entry}", structure={"empty": {}}
    )
    with pytest.raises(ValueError, match="empty"):
        list(getattr(species, entry)())
