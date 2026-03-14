"""
Genetic Simulation Utilities
============================

提供遗传学模拟的核心组件：结构、实体、种群模型。
"""

from natal.genetic_structures import Species, Chromosome, Locus
from natal.genetic_entities import Gene, Haplotype, HaploidGenotype, Genotype
from natal.index_core import IndexCore, compress_hg_glab, decompress_hg_glab
from natal.population_config import *
from natal.population_state import *
from natal.modifiers import *
from natal.type_def import *
from natal.algorithms import *
from natal.base_population import *
from natal.age_structured_population import *
from natal.population_builder import (
    PopulationBuilderBase,
    AgeStructuredPopulationBuilder, 
    DiscreteGenerationPopulationBuilder
)
from natal.recipes import GeneDriveRecipe, HomingModificationDrive, apply_recipe_to_population
from natal.hook_dsl import *

__version__ = "0.1.0"
__all__ = [
    # Genetic structures
    'Species', 'Chromosome', 'Locus',
    
    # Entities
    'Gene', 'Haplotype', 'HaploidGenotype', 'Genotype',
    
    # Indexing
    'IndexCore', 'compress_hg_glab', 'decompress_hg_glab',
    
    # Population & Configuration
    'BasePopulation', 'AgeStructuredPopulation',
    'AgeStructuredInitConfig', 'DiscreteGenerationInitConfig',
    
    # Builders
    'PopulationBuilderBase',
    'AgeStructuredPopulationBuilder',
    'DiscreteGenerationPopulationBuilder',
    
    # Recipes & Modifiers
    'GeneDriveRecipe', 'HomingModificationDrive', 'apply_recipe_to_population',
    'GameteModifier', 'ZygoteModifier',
]
