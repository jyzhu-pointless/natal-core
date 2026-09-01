"""Entities subpackage — runtime genetic entity classes.

Provides the concrete entity types (Gene, Haplotype, HaploidGenotype,
Genotype) that are bound to genetic structures (Locus, Chromosome, Species)
at runtime.  Entities auto-register to their parent structure on creation
and are cached so the same name under the same structure resolves to one
instance.
"""
