"""
Numba Switchable Decorators
===========================

Provides configurable Numba JIT compilation for functions (@njit_switch) and
classes (@jitclass_switch) with a single global control switch.
"""

import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Optional,
    ParamSpec,
    TypedDict,
    TypeVar,
    cast,
    overload,
)

__all__ = [
    "NUMBA_ENABLED", "is_numba_enabled", "enable_numba", "disable_numba",
    "NUMBA_LOG_ENABLED", "is_numba_log_enabled", "enable_numba_log", "disable_numba_log",
    "NUMBA_SIGNATURE_TRACE_ENABLED", "is_numba_signature_trace_enabled",
    "enable_numba_signature_trace", "disable_numba_signature_trace",
    "NUMBA_CACHE_DIR", "get_numba_cache_dir",
    "njit_switch", "numba_disabled", "numba_enabled",
    "with_numba_disabled", "with_numba_enabled"
]

# ============================================================================
# Global Configuration
# ============================================================================

# Master switch to enable/disable all Numba JIT compilation
# Default: True (Numba is ENABLED by default for performance)
# Set to False to use pure Python (useful for debugging)
NUMBA_ENABLED: bool = True

# Master switch for all JIT/cache status output (default: ON)
NUMBA_LOG_ENABLED: bool = True

# Optional diagnostics for new JIT specialization signatures.
NUMBA_SIGNATURE_TRACE_ENABLED: bool = False

# Default cache directory: project root /.numba_cache
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
NUMBA_CACHE_DIR: str = os.environ.get("NUMBA_CACHE_DIR", str(_PROJECT_ROOT / ".numba_cache"))

_NUMBA_CACHE_LOG_PATCHED = False
_NUMBA_CACHE_LOG_LOCK = threading.Lock()
_NUMBA_DISPATCHER_PATCHED = False
_NUMBA_DISPATCHER_LOCK = threading.Lock()
_NUMBA_COMPILE_CONTEXT = threading.local()
_NUMBA_LOG_IO_LOCK = threading.Lock()
_CACHE_FUNC_RE = re.compile(r"\.([^.\\/]+)-\d+\.py\d+\.")

P = ParamSpec("P")
R = TypeVar("R")


# TypedDict for compile feedback tracking
class CompileFeedback(TypedDict, total=False):
    """Feedback dict for compile progress tracking."""
    fn_name: str
    prefix: str
    is_tty: bool
    is_nested: bool
    depth: int
    stop_event: Optional[threading.Event]
    thread: Optional[threading.Thread]
    spinner_active: bool
    frozen_for_nested: bool
    cache_hit: bool
    cache_stored: bool
    parent_feedback: Optional["CompileFeedback"]
    feedback: Optional["CompileFeedback"]
    seen_child_functions: set[str]


# TypedDict for compile context in stack
class CompileContext(TypedDict, total=False):
    """Context for tracking compilation in progress."""
    fn_name: str
    cache_hit: bool
    cache_stored: bool
    feedback: Optional[CompileFeedback]
    before_sigs: tuple[str, ...]
    seen_child_functions: set[str]


def get_numba_cache_dir() -> str:
    """Return the active Numba cache directory."""
    return NUMBA_CACHE_DIR


def _log_print(*args: Any, **kwargs: Any) -> None:
    """Thread-safe print for interleaved spinner/cache output."""
    with _NUMBA_LOG_IO_LOCK:
        print(*args, **kwargs)


def _freeze_spinner_for_nested_output(feedback: Optional[CompileFeedback]) -> None:
    """Stop active top-level spinner and place cursor on a clean line."""
    if feedback is None or not feedback.get("is_tty"):
        return
    if not feedback.get("spinner_active", False):
        return

    stop_event = feedback.get("stop_event")
    thread = feedback.get("thread")
    prefix = feedback.get("prefix", "💡 Compiling function...")
    if stop_event is not None:
        stop_event.set()
    if thread is not None:
        thread.join(timeout=0.2)

    with _NUMBA_LOG_IO_LOCK:
        print("\r" + " " * (len(prefix) + 4) + "\r", end="", flush=True)
        print(prefix, flush=True)

    feedback["spinner_active"] = False
    feedback["frozen_for_nested"] = True


def _append_or_print(feedback: Optional[CompileFeedback], line: str) -> None:
    """Print child logs immediately (real-time)."""
    _log_print(line, flush=True)


def _apply_numba_cache_dir() -> None:
    """Apply default cache dir to environment and Numba config."""
    cache_dir = Path(NUMBA_CACHE_DIR)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)

    try:
        from numba.core import config as numba_config  # pyright: ignore
        config_obj: Any = numba_config
        setattr(config_obj, "CACHE_DIR", str(cache_dir))  # noqa: B010
    except Exception:
        pass


_apply_numba_cache_dir()


def _start_compile_feedback(fn_name: str) -> Optional[CompileFeedback]:
    """Emit immediate compile start feedback and animate a TTY spinner on one line."""
    if not NUMBA_LOG_ENABLED:
        return None

    is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    if not is_tty:
        # Non-interactive output falls back to plain two-line logs.
        _log_print(f"💡 Compiling function: `{fn_name}`...", flush=True)
        return {"fn_name": fn_name, "is_tty": False}

    stop_event = threading.Event()
    prefix = f"💡 Compiling function: `{fn_name}`..."
    spinner_frames = "|/-\\"

    def _animate_progress() -> None:
        frame_index = 0
        while not stop_event.wait(0.08):
            frame = spinner_frames[frame_index % len(spinner_frames)]
            with _NUMBA_LOG_IO_LOCK:
                print(f"\r{prefix} {frame}", end="", flush=True)
            frame_index += 1

    with _NUMBA_LOG_IO_LOCK:
        print(f"\r{prefix} {spinner_frames[0]}", end="", flush=True)

    thread = threading.Thread(target=_animate_progress, daemon=True)
    thread.start()

    return {
        "fn_name": fn_name,
        "prefix": prefix,
        "is_tty": True,
        "stop_event": stop_event,
        "thread": thread,
        "is_nested": False,
        "spinner_active": True,
        "frozen_for_nested": False,
    }


def _start_nested_compile_feedback(
    fn_name: str,
    depth: int,
    parent_feedback: Optional[CompileFeedback] = None,
) -> Optional[CompileFeedback]:
    """Emit start feedback for nested JIT compilation calls (same style as top-level)."""
    if not NUMBA_LOG_ENABLED:
        return None

    _freeze_spinner_for_nested_output(parent_feedback)

    indent = "  " * max(depth - 1, 0)
    prefix = f"{indent}💡 Compiling function: `{fn_name}`..."

    return {
        "fn_name": fn_name,
        "prefix": prefix,
        "is_tty": False,
        "is_nested": True,
        "depth": depth,
        "parent_feedback": parent_feedback,
    }


def _finish_compile_feedback(feedback: Optional[CompileFeedback], cached: bool, elapsed: float) -> None:
    """Finalize compile feedback with cached/done result."""
    if not NUMBA_LOG_ENABLED or feedback is None:
        return

    fn_name = feedback.get("fn_name", "<unknown>")
    if feedback.get("is_nested"):
        status = f"⚡️ Done in {elapsed:.2f} s (cache hit)" if cached else f"✅ Done in {elapsed:.2f} s"
        prefix = feedback.get("prefix", f"💡 Compiling function: `{fn_name}`...")
        _log_print(f"{prefix} {status}", flush=True)
        return

    if feedback.get("is_tty"):
        prefix = feedback.get("prefix", f"💡 Compiling function: `{fn_name}`...")
        status = f"⚡️ Done in {elapsed:.2f} s (cache hit)" if cached else f"✅ Done in {elapsed:.2f} s"

        if feedback.get("spinner_active", False):
            stop_event = feedback.get("stop_event")
            thread = feedback.get("thread")
            if stop_event is not None:
                stop_event.set()
            if thread is not None:
                thread.join(timeout=0.2)
            with _NUMBA_LOG_IO_LOCK:
                clear_width = len(prefix) + len(status)
                print("\r" + " " * clear_width + "\r", end="", flush=True)
                print(f"{prefix} {status}", flush=True)
        else:
            _log_print(f"{prefix} {status}", flush=True)
        return

    if cached:
        _log_print(f"⚡️ Done in {elapsed:.2f} s (cache hit)", flush=True)
    else:
        _log_print(f"✅ Done in {elapsed:.2f} s", flush=True)


def _get_compile_context_stack() -> list[CompileContext]:
    """Get per-thread compile context stack for associating cache events."""
    stack = cast(Optional[list[CompileContext]], getattr(_NUMBA_COMPILE_CONTEXT, "stack", None))
    if stack is None:
        stack = []
        _NUMBA_COMPILE_CONTEXT.stack = stack
    return stack


def _extract_cached_function_name(rendered_cache_msg: str) -> Optional[str]:
    """Extract function name from Numba cache path message if possible."""
    match = _CACHE_FUNC_RE.search(rendered_cache_msg)
    if not match:
        return None
    return match.group(1)


def _emit_signature_trace(fn_name: str, depth: int, new_sigs: list[str], args: tuple[Any, ...]) -> None:
    """Emit optional diagnostics for newly created JIT specializations."""
    if not NUMBA_SIGNATURE_TRACE_ENABLED:
        return

    indent = "  " * max(depth - 1, 0)
    arg_types = ", ".join(type(arg).__name__ for arg in args)
    if not arg_types:
        arg_types = "<none>"

    for sig_text in new_sigs:
        _log_print(
            f"{indent}🔎 Signature trace: {fn_name} | args=[{arg_types}] | sig={sig_text}",
            flush=True,
        )


def _install_cache_log_formatter() -> None:
    """Patch numba.core.caching._cache_log once to emit project-formatted messages."""
    global _NUMBA_CACHE_LOG_PATCHED

    if _NUMBA_CACHE_LOG_PATCHED:
        return

    with _NUMBA_CACHE_LOG_LOCK:
        if _NUMBA_CACHE_LOG_PATCHED:
            return

        try:
            from numba.core import caching as numba_caching  # pyright: ignore
        except Exception:
            return

        caching_obj: Any = numba_caching
        if not hasattr(caching_obj, "_cache_log"):
            return

        original_cache_log = cast(Callable[..., Any], getattr(caching_obj, "_cache_log"))  # noqa: B009

        @wraps(original_cache_log)
        def _formatted_cache_log(msg: Any, *args: Any) -> None:
            if not NUMBA_LOG_ENABLED:
                return

            try:
                rendered = (msg % args) if args else str(msg)
            except Exception:
                rendered = str(msg)

            normalized = rendered.lower()
            stack = _get_compile_context_stack()
            active_context = stack[-1] if stack else None
            depth = len(stack)
            child_indent = "  " * max(depth, 0)
            cached_fn = _extract_cached_function_name(rendered)

            if "data loaded from" in normalized:
                if active_context is not None:
                    active_context["cache_hit"] = True
                    parent_name = active_context.get("fn_name")
                    if cached_fn is not None and cached_fn != parent_name:
                        seen = active_context.setdefault("seen_child_functions", set())
                        if cached_fn not in seen:
                            seen.add(cached_fn)
                            parent_feedback = active_context.get("feedback")
                            _freeze_spinner_for_nested_output(parent_feedback)
                            _append_or_print(
                                parent_feedback,
                                f"{child_indent}💡 Compiling function: `{cached_fn}`... ⚡️ Cached",
                            )
            elif "data saved to" in normalized:
                if active_context is not None:
                    active_context["cache_stored"] = True
                    parent_name = active_context.get("fn_name")
                    if cached_fn is not None and cached_fn != parent_name:
                        seen = active_context.setdefault("seen_child_functions", set())
                        if cached_fn not in seen:
                            seen.add(cached_fn)
                            parent_feedback = active_context.get("feedback")
                            _freeze_spinner_for_nested_output(parent_feedback)
                            _append_or_print(
                                parent_feedback,
                                f"{child_indent}💡 Compiling function: `{cached_fn}`... ✅ Cache Stored",
                            )

        setattr(caching_obj, "_cache_log", _formatted_cache_log)  # noqa: B010
        _NUMBA_CACHE_LOG_PATCHED = True  # pyright: ignore[reportConstantRedefinition]


def _install_dispatcher_compile_formatter() -> None:
    """Patch Dispatcher._compile_for_args once to log JIT specialization events."""
    global _NUMBA_DISPATCHER_PATCHED

    if _NUMBA_DISPATCHER_PATCHED:
        return

    with _NUMBA_DISPATCHER_LOCK:
        if _NUMBA_DISPATCHER_PATCHED:
            return

        try:
            from numba.core import dispatcher as numba_dispatcher  # pyright: ignore
        except Exception:
            return

        if not hasattr(numba_dispatcher, "Dispatcher"):
            return

        dispatcher_cls = numba_dispatcher.Dispatcher
        dispatcher_obj: Any = dispatcher_cls
        original_compile_for_args = cast(Callable[..., Any], getattr(dispatcher_obj, "_compile_for_args"))  # noqa: B009

        @wraps(original_compile_for_args)
        def _formatted_compile_for_args(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not NUMBA_LOG_ENABLED:
                return original_compile_for_args(self, *args, **kwargs)

            fn_name = getattr(getattr(self, "py_func", None), "__name__", "<unknown>")
            start_time = time.perf_counter()
            stack = _get_compile_context_stack()
            compile_depth = len(stack)

            if compile_depth == 0:
                feedback = _start_compile_feedback(fn_name)
                root_seen_child_functions: set[str] = set()
            else:
                root_context = stack[0] if stack else None
                root_seen_child_functions = cast(set[str], root_context.get("seen_child_functions", set())) if root_context else set()
                should_log_nested = fn_name not in root_seen_child_functions
                if should_log_nested:
                    root_seen_child_functions.add(fn_name)
                parent_feedback = stack[-1].get("feedback") if stack else None
                feedback = _start_nested_compile_feedback(fn_name, compile_depth, parent_feedback) if should_log_nested else None

            context: CompileContext = {
                "fn_name": fn_name,
                "cache_hit": False,
                "cache_stored": False,
                "feedback": feedback,
                "before_sigs": tuple(str(sig) for sig in getattr(self, "signatures", ())),
                "seen_child_functions": root_seen_child_functions,
            }
            stack.append(context)
            result: Any
            try:
                result = original_compile_for_args(self, *args, **kwargs)
            finally:
                stack.pop()

            elapsed = time.perf_counter() - start_time
            after_sigs = tuple(str(sig) for sig in getattr(self, "signatures", ()))
            before_sig_set = set(context["before_sigs"])
            new_sigs = [sig for sig in after_sigs if sig not in before_sig_set]
            _emit_signature_trace(fn_name, compile_depth, new_sigs, args)
            _finish_compile_feedback(feedback, cached=context["cache_hit"], elapsed=elapsed)

            return result

        setattr(dispatcher_obj, "_compile_for_args", _formatted_compile_for_args)  # noqa: B010
        _NUMBA_DISPATCHER_PATCHED = True  # pyright: ignore[reportConstantRedefinition]


def is_numba_enabled() -> bool:
    """
    Check if Numba JIT compilation is currently enabled.

    Returns:
        bool: True if Numba is enabled (default), False if disabled
    """
    return NUMBA_ENABLED


def is_numba_log_enabled() -> bool:
    """Check if Numba JIT/cache status logging is enabled."""
    return NUMBA_LOG_ENABLED


def is_numba_signature_trace_enabled() -> bool:
    """Check if specialization-signature trace logging is enabled."""
    return NUMBA_SIGNATURE_TRACE_ENABLED

# ============================================================================
# @njit_switch Decorator (Function-level)
# ============================================================================

@overload
def njit_switch(
    func: Callable[P, R],
    *,
    cache: bool = True,
    parallel: bool = False,
    fastmath: bool = False,
    **njit_kwargs: Any,
) -> Callable[P, R]:
    ...


@overload
def njit_switch(
    func: None = None,
    *,
    cache: bool = True,
    parallel: bool = False,
    fastmath: bool = False,
    **njit_kwargs: Any,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    ...


def njit_switch(
    func: Optional[Callable[P, R]] = None,
    *,
    cache: bool = True,
    parallel: bool = False,
    fastmath: bool = False,
    **njit_kwargs: Any,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
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

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        numba_func: Optional[Callable[P, R]] = None

        if NUMBA_ENABLED:
            try:
                from numba import njit  # pyright: ignore
                from numba.core import config as numba_config  # pyright: ignore

                _apply_numba_cache_dir()

                # Keep all JIT/cache output under one project switch.
                config_obj: Any = numba_config
                setattr(config_obj, "DEBUG_CACHE", 1 if NUMBA_LOG_ENABLED else 0)  # noqa: B010
                _install_cache_log_formatter()
                _install_dispatcher_compile_formatter()

                numba_func = cast(
                    Callable[P, R],
                    njit(
                        fn,
                        cache=cache,
                        parallel=parallel,
                        fastmath=fastmath,
                        **njit_kwargs,
                    ),
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
# Utility Functions
# ============================================================================

def disable_numba():
    """Disable Numba JIT compilation globally (use pure Python).

    Useful for debugging or when you need Python-compatible error messages.
    Note: Numba is ENABLED by default.
    """
    global NUMBA_ENABLED
    NUMBA_ENABLED = False  # pyright: ignore[reportConstantRedefinition]


def disable_numba_log():
    """Disable all formatted Numba JIT/cache status output."""
    global NUMBA_LOG_ENABLED
    NUMBA_LOG_ENABLED = False  # pyright: ignore[reportConstantRedefinition]

    try:
        from numba.core import config as numba_config  # pyright: ignore
        config_obj: Any = numba_config
        setattr(config_obj, "DEBUG_CACHE", 0)  # noqa: B010
    except Exception:
        pass


def disable_numba_signature_trace():
    """Disable signature diagnostics for newly created Numba specializations."""
    global NUMBA_SIGNATURE_TRACE_ENABLED
    NUMBA_SIGNATURE_TRACE_ENABLED = False  # pyright: ignore[reportConstantRedefinition]


def enable_numba():
    """Re-enable Numba JIT compilation globally (default state).

    Numba is enabled by default for maximum performance.
    Call this to restore Numba after it was disabled.
    """
    global NUMBA_ENABLED
    NUMBA_ENABLED = True  # pyright: ignore[reportConstantRedefinition]
    _apply_numba_cache_dir()


def enable_numba_log():
    """Enable formatted Numba JIT/cache status output (default state)."""
    global NUMBA_LOG_ENABLED
    NUMBA_LOG_ENABLED = True  # pyright: ignore[reportConstantRedefinition]

    try:
        from numba.core import config as numba_config  # pyright: ignore
        config_obj: Any = numba_config
        setattr(config_obj, "DEBUG_CACHE", 1)  # noqa: B010
        _install_cache_log_formatter()
        _install_dispatcher_compile_formatter()
    except Exception:
        pass


def enable_numba_signature_trace():
    """Enable signature diagnostics for newly created Numba specializations."""
    global NUMBA_SIGNATURE_TRACE_ENABLED
    NUMBA_SIGNATURE_TRACE_ENABLED = True  # pyright: ignore[reportConstantRedefinition]


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
    NUMBA_ENABLED = False  # pyright: ignore[reportConstantRedefinition]
    try:
        yield
    finally:
        NUMBA_ENABLED = original_state  # pyright: ignore[reportConstantRedefinition]


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
    NUMBA_ENABLED = True  # pyright: ignore[reportConstantRedefinition]
    try:
        yield
    finally:
        NUMBA_ENABLED = original_state  # pyright: ignore[reportConstantRedefinition]


def with_numba_disabled(func: Callable[P, R]) -> Callable[P, R]:
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
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with numba_disabled():
            return func(*args, **kwargs)
    return wrapper


def with_numba_enabled(func: Callable[P, R]) -> Callable[P, R]:
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
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with numba_enabled():
            return func(*args, **kwargs)
    return wrapper
