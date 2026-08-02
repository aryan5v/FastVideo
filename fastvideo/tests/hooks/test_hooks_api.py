from fastvideo.hooks.hooks import ForwardHook, ModuleHookManager
from torch import nn
from typing import Any
import pytest
import torch


class EventHook(ForwardHook):
    def __init__(self, content: str, event_list: list[str]):
        self.content = content
        self.event_list = event_list

    def name(self) -> str:
        return f"EventHook_{self.content}"

    def pre_forward(self, module: nn.Module, *args, **kwargs):
        print(
            f"[{self.content}] Pre-forward called with args[0].shape: {args[0].shape}"
        )
        self.event_list.append(f"[pre]{self.content}")
        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any):
        print(
            f"[{self.content}] Post-forward called with outputs.shape: {output.shape}"
        )
        self.event_list.append(f"[post]{self.content}")
        return output


def test_hook_execution_order():
    """Test that hooks are executed in the correct order: LIFO for pre-hooks, FIFO for post-hooks."""
    # Create a simple model
    model = nn.Linear(10, 20)

    # Create event list to track hook execution order
    events = []

    # Create and push hooks in order: A then B

    manager = ModuleHookManager.get_from_or_default(model)

    hook_a = EventHook("A", events)
    hook_b = EventHook("B", events)

    manager.append_forward_hook(hook_a)
    manager.append_forward_hook(hook_b)

    # Perform a forward pass
    input_tensor = torch.randn(2, 10)
    model(input_tensor)

    # Verify the execution order is [pre_a, pre_b, post_b, post_a]
    # Pre-hooks should be FILO (First In Last Out): A then B
    # Post-hooks should be LIFO (Last In First Out): B then A
    expected_events = ["[pre]A", "[pre]B", "[post]B", "[post]A"]

    assert events == expected_events, (
        f"Expected {expected_events}, but got {events}"
    )
    print(f"✓ Hook execution order test passed: {events}")


# ---------------------------------------------------------------------------
# Exception-safe unwinding (PR #26 review)
#
# These hooks carry real state -- parameter offload and materialization -- so a
# pre_forward that ran without its post_forward leaves the module
# half-materialized, and that survives the exception into every later call.
# Artifact dispatch makes this reachable by design: an untrusted candidate may
# raise and be replaced by the native forward.
# ---------------------------------------------------------------------------


class _CountingHook(ForwardHook):
    """Records pre/post calls so the lifecycle can be checked for balance."""

    def __init__(self, name: str, events: list[str]) -> None:
        self._name = name
        self.events = events

    def name(self) -> str:  # type: ignore[override]
        return self._name

    def pre_forward(self, module, *args, **kwargs):
        self.events.append(f"pre:{self._name}")
        return args, kwargs

    def post_forward(self, module, output):
        self.events.append(f"post:{self._name}")
        return output


def test_post_forward_runs_when_the_forward_raises():
    events: list[str] = []
    module = torch.nn.Linear(2, 2)
    manager = ModuleHookManager.get_from_or_default(module)
    manager.append_forward_hook(_CountingHook("a", events))
    manager.append_forward_hook(_CountingHook("b", events))

    def boom(*args, **kwargs):
        raise RuntimeError("candidate exploded")

    with pytest.raises(RuntimeError, match="candidate exploded"):
        manager.run_with_forward(boom, torch.zeros(1, 2))

    assert events.count("pre:a") == events.count("post:a") == 1
    assert events.count("pre:b") == events.count("post:b") == 1
    # Unwound newest-first, mirroring the successful path.
    assert events == ["pre:a", "pre:b", "post:b", "post:a"]
    ModuleHookManager.remove_from_manager(module)


def test_only_completed_pre_hooks_are_unwound():
    """A hook whose pre_forward raised never ran, so its post must not run."""
    events: list[str] = []
    module = torch.nn.Linear(2, 2)
    manager = ModuleHookManager.get_from_or_default(module)

    class _RaisingPre(_CountingHook):
        def pre_forward(self, module, *args, **kwargs):
            self.events.append(f"pre:{self._name}")
            raise ValueError("pre-hook failed")

    manager.append_forward_hook(_CountingHook("a", events))
    manager.append_forward_hook(_RaisingPre("b", events))
    manager.append_forward_hook(_CountingHook("c", events))

    with pytest.raises(ValueError, match="pre-hook failed"):
        manager.run_with_forward(lambda *a, **k: None, torch.zeros(1, 2))

    assert events == ["pre:a", "pre:b", "post:a"]
    assert "post:b" not in events, "a pre-hook that raised must not be unwound"
    assert "pre:c" not in events, "hooks after the failure never ran"
    ModuleHookManager.remove_from_manager(module)


def test_a_failing_post_hook_does_not_mask_the_original_error():
    events: list[str] = []
    module = torch.nn.Linear(2, 2)
    manager = ModuleHookManager.get_from_or_default(module)

    class _RaisingPost(_CountingHook):
        def post_forward(self, module, output):
            self.events.append(f"post:{self._name}")
            raise ValueError("unwind failure")

    manager.append_forward_hook(_RaisingPost("a", events))

    with pytest.raises(RuntimeError, match="original"):
        manager.run_with_forward(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("original")))

    assert events == ["pre:a", "post:a"]
    ModuleHookManager.remove_from_manager(module)


def test_successful_forward_still_runs_hooks_in_reverse_order():
    events: list[str] = []
    module = torch.nn.Linear(2, 2)
    manager = ModuleHookManager.get_from_or_default(module)
    manager.append_forward_hook(_CountingHook("a", events))
    manager.append_forward_hook(_CountingHook("b", events))

    manager.run_with_forward(lambda *a, **k: "out", torch.zeros(1, 2))

    assert events == ["pre:a", "pre:b", "post:b", "post:a"]
    ModuleHookManager.remove_from_manager(module)
