"""
Numba Switchable Decorators
===========================

Provides configurable Numba JIT compilation for functions (@njit_switch) and 
classes (@jitclass_switch) with a single global control switch.
"""

from typing import Callable, Optional
from contextlib import contextmanager

__all__ = [
    "NUMBA_ENABLED", "is_numba_enabled", "enable_numba", "disable_numba",
    "jitclass_switch", "njit_switch", "numba_disabled", "numba_enabled",
    "with_numba_disabled", "with_numba_enabled"
]

# ============================================================================
# Global Configuration
# ============================================================================

# Master switch to enable/disable all Numba JIT compilation
# Default: True (Numba is ENABLED by default for performance)
# Set to False to use pure Python (useful for debugging)
NUMBA_ENABLED: bool = True


def is_numba_enabled() -> bool:
    """
    Check if Numba JIT compilation is currently enabled.
    
    Returns:
        bool: True if Numba is enabled (default), False if disabled
    """
    return NUMBA_ENABLED

# ============================================================================
# @njit_switch Decorator (Function-level)
# ============================================================================

def njit_switch(
    func: Optional[Callable] = None,
    *,
    cache: bool = True,
    parallel: bool = False,
    fastmath: bool = False,
    **njit_kwargs
) -> Callable:
    """
    Numba @njit decorator for functions (controlled by global NUMBA_ENABLED flag).
    
    Args:
        func: Function to decorate
        cache: Cache compiled functions (default: True)
        parallel: Enable automatic parallelization (default: False)
        fastmath: Enable fast math optimizations (default: False)
        **njit_kwargs: Additional arguments for numba.njit
    
    Usage:
        ```python
        @njit_switch
        def my_func(x):
            ...
        
        @njit_switch(parallel=True, fastmath=True)
        def my_parallel_func(x):
            ...
        ```
    """
    
    def decorator(fn: Callable) -> Callable:
        numba_func = None
        
        if NUMBA_ENABLED:
            try:
                from numba import njit
                numba_func = njit(
                    fn,
                    cache=cache,
                    parallel=parallel,
                    fastmath=fastmath,
                    **njit_kwargs
                )
            except ImportError:
                pass
            except Exception:
                pass
        
        # Return compiled or original function
        return numba_func if numba_func is not None else fn
    
    # Support both @njit_switch and @njit_switch(...)
    if func is not None:
        return decorator(func)
    return decorator

# ============================================================================
# @jitclass_switch Decorator (Class-level)
# ============================================================================

def jitclass_switch(
    spec_or_cls: Optional[type | list] = None,
    **jitclass_kwargs
) -> Callable:
    """
    Numba @jitclass decorator for classes (controlled by global NUMBA_ENABLED flag).
    
    Args:
        spec_or_cls: Either a spec list (when used as @jitclass_switch(spec))
                     or the class itself (when used as @jitclass_switch)
        **jitclass_kwargs: Additional arguments for numba.jitclass
    
    Usage:
        ```python
        # With spec (recommended)
        spec = [('field1', types.int64), ('field2', types.float64)]
        @jitclass_switch(spec)
        class MyClass:
            ...
        
        # Without spec (inference not always possible)
        @jitclass_switch
        class MyClass:
            ...
        ```
    """
    
    def decorator(cls: type) -> type:
        numba_cls = None
        
        if NUMBA_ENABLED:
            try:
                from numba.experimental import jitclass as numba_jitclass
                numba_cls = numba_jitclass(spec_or_cls, **jitclass_kwargs)(cls)
            except ImportError:
                pass
            except Exception:
                pass
        
        # Return compiled or original class
        return numba_cls if numba_cls is not None else cls
    
    # Support both @jitclass_switch and @jitclass_switch(spec)
    if spec_or_cls is not None and isinstance(spec_or_cls, type):
        # Used as @jitclass_switch (spec_or_cls is the class)
        return decorator(spec_or_cls)
    else:
        # Used as @jitclass_switch(spec) (spec_or_cls is the spec)
        # Return decorator that will be applied to the class
        def spec_decorator(cls: type) -> type:
            return decorator(cls)
        return spec_decorator

# ============================================================================
# Utility Functions
# ============================================================================

def disable_numba():
    """Disable Numba JIT compilation globally (use pure Python).
    
    Useful for debugging or when you need Python-compatible error messages.
    Note: Numba is ENABLED by default.
    """
    global NUMBA_ENABLED
    NUMBA_ENABLED = False


def enable_numba():
    """Re-enable Numba JIT compilation globally (default state).
    
    Numba is enabled by default for maximum performance.
    Call this to restore Numba after it was disabled.
    """
    global NUMBA_ENABLED
    NUMBA_ENABLED = True


@contextmanager
def numba_disabled():
    """Context manager to temporarily disable Numba (use pure Python).
    
    Useful for debugging within a specific block. Numba is enabled by default,
    so it will be automatically restored after exiting the context.
    
    Usage:
        ```python
        from natal.numba_utils import numba_disabled
        
        # Default: Numba enabled
        with numba_disabled():
            pop.run(n_steps=10)  # Temporarily uses pure Python
        # Numba automatically re-enabled here
        ```
    """
    global NUMBA_ENABLED
    original_state = NUMBA_ENABLED
    NUMBA_ENABLED = False
    try:
        yield
    finally:
        NUMBA_ENABLED = original_state


@contextmanager
def numba_enabled():
    """Context manager to ensure Numba is enabled (default state).
    
    Numba is enabled by default, but this can force re-enable it if temporarily
    disabled elsewhere. The original state is restored after exiting the context.
    
    Usage:
        ```python
        from natal.numba_utils import numba_enabled
        
        with numba_enabled():
            pop.run(n_steps=100)  # Guaranteed to run with JIT
        # Original state is restored after
        ```
    """
    global NUMBA_ENABLED
    original_state = NUMBA_ENABLED
    NUMBA_ENABLED = True
    try:
        yield
    finally:
        NUMBA_ENABLED = original_state


def with_numba_disabled(func: Callable) -> Callable:
    """Decorator to run a function with Numba disabled (pure Python).
    
    Useful for testing functions with pure Python, which provides better
    error messages and debugging capabilities. Numba is enabled by default,
    so this decorator will restore it after the function returns.
    
    Usage:
        ```python
        from natal.numba_utils import with_numba_disabled
        
        @with_numba_disabled
        def debug_version():
            pop.run(n_steps=10)
            # Runs with pure Python for easier debugging
        
        debug_version()  # Temporarily disables Numba
        # Numba automatically restored after
        ```
    """
    def wrapper(*args, **kwargs):
        with numba_disabled():
            return func(*args, **kwargs)
    return wrapper


def with_numba_enabled(func: Callable) -> Callable:
    """Decorator to ensure a function runs with Numba enabled (default state).
    
    Since Numba is enabled by default, this is primarily useful for ensuring
    a function uses JIT even if Numba was temporarily disabled elsewhere.
    
    Usage:
        ```python
        from natal.numba_utils import with_numba_enabled
        
        @with_numba_enabled
        def performance_critical():
            pop.run(n_steps=1000)
            # Guaranteed to run with JIT for maximum speed
        
        performance_critical()  # Numba guaranteed enabled
        # Original state is restored after
        ```
    """
    def wrapper(*args, **kwargs):
        with numba_enabled():
            return func(*args, **kwargs)
    return wrapper
