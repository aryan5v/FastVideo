# SPDX-License-Identifier: Apache-2.0

import functools
from collections.abc import Callable
from typing import Any

from torch import nn

from fastvideo.logger import init_logger

logger = init_logger(__name__)


class ForwardHook:
    """
    Base class for forward hooks.
    Hooks are used in the way:
        modified_args, modified_kwargs = hook.pre_forward(module, *args, **kwargs)
        output = module.forward(*modified_args, **modified_kwargs)
        modified_output = hook.post_forward(module, output)
    """

    @classmethod
    def name(cls) -> str:
        raise NotImplementedError

    def on_attach(self, module: nn.Module):  # noqa: B027
        """Called once when the hook is attached to the module."""
        pass

    def on_detach(self, module: nn.Module):  # noqa: B027
        """
        Called once when the hook is detached from the module.
        Note: this function is not guaranteed to be called if the module is
              deleted before the hook is detached.
        """
        pass

    def pre_forward(self, module: nn.Module, *args, **kwargs) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Called before the module's forward method is executed."""
        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        """Called after the module's forward method is executed."""
        return output


class ModuleHookManager:
    module_hook_attribute = "_hook_manager"

    def __init__(self, module: nn.Module):
        self.module = module
        self.forward_hooks: dict[str, ForwardHook] = {}
        self.original_forward = module.forward

    @classmethod
    def get_from(cls, module: nn.Module) -> "ModuleHookManager | None":
        if hasattr(module, cls.module_hook_attribute):
            return getattr(module, cls.module_hook_attribute)
        return None

    @classmethod
    def get_from_or_default(cls, module: nn.Module) -> "ModuleHookManager":
        if not hasattr(module, cls.module_hook_attribute):
            setattr(module, cls.module_hook_attribute, cls(module))

            def forward_hook_wrapper(mod: nn.Module, *args, **kwargs):
                manager: ModuleHookManager = getattr(mod, cls.module_hook_attribute)
                return manager.run_with_forward(
                    manager.original_forward,
                    *args,
                    **kwargs,
                )

            module.forward = functools.partial(forward_hook_wrapper, module)

        return getattr(module, cls.module_hook_attribute)

    def run_with_forward(
        self,
        forward: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run ``forward`` through this module's installed hook lifecycle.

        The lifecycle is balanced even when something raises. Hooks here carry
        real state -- parameter offload and materialization -- so a pre-hook
        that ran without its post-hook leaves the module in a half-materialized
        state that survives the exception and corrupts every later call. That
        matters most for artifact dispatch, whose whole design is that an
        untrusted candidate may raise and be replaced by the native forward.

        On failure the post-hooks for exactly the pre-hooks that completed are
        run, in reverse order, and the original exception is re-raised. A
        post-hook that itself fails during unwinding is logged and does not
        mask the original error.
        """
        completed: list[ForwardHook] = []
        try:
            for hook in self.forward_hooks.values():
                args, kwargs = hook.pre_forward(self.module, *args, **kwargs)
                completed.append(hook)
            output = forward(*args, **kwargs)
        except Exception:
            self._unwind(completed)
            raise
        for hook in reversed(completed):
            output = hook.post_forward(self.module, output)
        return output

    def _unwind(self, completed: list["ForwardHook"]) -> None:
        """Run post-hooks for the pre-hooks that completed, newest first."""
        for hook in reversed(completed):
            try:
                hook.post_forward(self.module, None)
            except Exception:  # noqa: BLE001 - never mask the original failure
                logger.warning(
                    "post_forward failed while unwinding %s on %s; continuing "
                    "so the original error is not masked",
                    type(hook).__name__,
                    type(self.module).__name__,
                    exc_info=True,
                )

    @staticmethod
    def remove_from_manager(module: nn.Module) -> None:
        if hasattr(module, ModuleHookManager.module_hook_attribute):
            manager: ModuleHookManager = getattr(module, ModuleHookManager.module_hook_attribute)
            module.forward = manager.original_forward
            delattr(module, ModuleHookManager.module_hook_attribute)

    def _check_manager_attached(self) -> None:
        if not hasattr(self.module, self.module_hook_attribute):
            raise ValueError("ModuleHookManager is not attached to the module.")
        if getattr(self.module, self.module_hook_attribute) is not self:
            raise ValueError("ModuleHookManager attached to the module is different.")

    def append_forward_hook(self, hook: ForwardHook):
        self._check_manager_attached()
        if hook.name() in self.forward_hooks:
            raise ValueError(f"Hook with name {hook.name()} is already registered.")
        # after python 3.7, dicts maintain insertion order
        self.forward_hooks[hook.name()] = hook
        hook.on_attach(self.module)

    def replace_forward_hook(self, hook_name: str, new_hook: ForwardHook, run_on_attach: bool = True):
        self._check_manager_attached()
        if hook_name not in self.forward_hooks:
            raise ValueError(f"No hook with name {hook_name} found.")
        old_hook = self.forward_hooks[hook_name]
        if run_on_attach:
            old_hook.on_detach(self.module)
        self.forward_hooks[hook_name] = new_hook
        new_hook.on_attach(self.module)

    def remove_forward_hook(self, hook_name: str, run_detach: bool = True):
        self._check_manager_attached()
        if hook_name not in self.forward_hooks:
            raise ValueError(f"No hook with name {hook_name} found.")
        if run_detach:
            self.forward_hooks[hook_name].on_detach(self.module)
        del self.forward_hooks[hook_name]

    def get_forward_hook(self, hook_name: str) -> ForwardHook | None:
        return self.forward_hooks.get(hook_name, None)
