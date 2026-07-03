"""Builder functions — split across _plain, _config, _engine modules."""
from natal.data._config import build_population_config
from natal.data._engine import (
    build_custom_array,
    build_discrete_engine_config,
    compress_config,
    compress_hl,
    decompress_hl,
    initialize_gamete_map,
    initialize_zygote_map,
)
from natal.data._plain import (
    from_plain_population_config,
    to_plain_population_config,
)

__all__ = [
    'build_population_config',
    'build_discrete_engine_config',
    'build_custom_array',
    'initialize_zygote_map',
    'initialize_gamete_map',
    'compress_hl',
    'decompress_hl',
    'compress_config',
    'to_plain_population_config',
    'from_plain_population_config',
]
