# mypy: allow-untyped-defs
from typing import cast, List, Optional, Union

import torch
from torch import Tensor

from torch.optim.optimizer import (
    _default_to_fused_or_foreach,
    _device_dtype_check_for_fused,
    _differentiable_doc,
    _foreach_doc,
    _fused_doc,
    _maximize_doc,
    _params_doc,
    _use_grad_for_differentiable,
    DeviceDict,
    Optimizer,
    ParamsT,
)


__all__ = ["SGD", "sgd"]


class SGD(Optimizer):  # noqa: D101
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):  # noqa: D107
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")


        defaults = dict(
            lr=lr,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            fused=fused,
        )
        super().__init__(params, defaults)

        if fused:
            self._step_supports_amp_scaling = True
            self._need_device_dtype_check_for_fused = True
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

    def __setstate__(self, state):  # noqa: D105
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)
            group.setdefault("fused", False)

    def _init_group(self, group, params, grads, momentum_buffer_list):
        has_sparse_grad = False

        for p in group["params"]:
            if p.grad is not None:
                if group["fused"] and getattr(
                    self, "_need_device_dtype_check_for_fused", True
                ):
                    _device_dtype_check_for_fused(p)
                    self._need_device_dtype_check_for_fused = False
                params.append(p)
                grads.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

        return has_sparse_grad

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []

            has_sparse_grad = self._init_group(
                group, params, grads
            )

            sgd(
                params,
                grads,
                lr=group["lr"],
                maximize=group["maximize"],
                has_sparse_grad=has_sparse_grad,
                foreach=group["foreach"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss

def sgd(
    params: List[Tensor],
    d_p_list: List[Tensor],
    has_sparse_grad: bool = False,
    foreach: Optional[bool] = None,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    *,
    lr: float,
    maximize: bool,
):
    if foreach is None and fused is None:
        if not torch.jit.is_scripting():
            fused, foreach = _default_to_fused_or_foreach(
                params, differentiable=False, use_fused=False
            )
        else:
            foreach = False
            fused = False
    if foreach is None:
        foreach = False
    if fused is None:
        fused = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sgd
    elif fused and not torch.jit.is_scripting():
        func = _fused_sgd
    else:
        func = _single_tensor_sgd

    func(
        params,
        d_p_list,
        lr=lr,
        has_sparse_grad=has_sparse_grad,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )


def _single_tensor_sgd(
    params: List[Tensor],
    grads: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    lr: float,
    maximize: bool,
    has_sparse_grad: bool,
):
    assert grad_scale is None and found_inf is None

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        param.add_(grad, alpha=-lr)


def _multi_tensor_sgd(
    params: List[Tensor],
    grads: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    lr: float,
    maximize: bool,
    has_sparse_grad: bool,
):
    assert grad_scale is None and found_inf is None

    if len(params) == 0:
        return

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads], with_indices=True  # type: ignore[list-item]
    )

    for (
        device_params_,
        device_grads_,
    ), indices in grouped_tensors.values():
        device_params: List[Tensor] = cast(List[Tensor], device_params_)
        device_grads: List[Tensor] = cast(List[Tensor], device_grads_)

        device_has_sparse_grad = has_sparse_grad and any(
            grad.is_sparse for grad in device_grads
        )

        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]


        if not device_has_sparse_grad:
            # handle internal item() call if lr is a tensor
            if isinstance(lr, torch.Tensor) and torch.compiler.is_compiling():
                grads_x_lr = torch._foreach_mul(device_grads, -lr)
                torch._foreach_add_(device_params, grads_x_lr)
            else:
                torch._foreach_add_(device_params, device_grads, alpha=-lr)
        else:
            # foreach APIs don't support sparse
            for i in range(len(device_params)):
                device_params[i].add_(device_grads[i], alpha=-lr)


def _fused_sgd(
    params: List[Tensor],
    grads: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    lr: float,
    maximize: bool,
    has_sparse_grad: bool,
) -> None:
    if not params:
        return
    if has_sparse_grad:
        raise RuntimeError("`_fused_sgd` does not support sparse gradients")
    grad_scale_dict: DeviceDict = (
        {grad_scale.device: grad_scale} if grad_scale is not None else {}
    )
    found_inf_dict: DeviceDict = (
        {found_inf.device: found_inf} if found_inf is not None else {}
    )

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads], with_indices=False  # type: ignore[list-item]
    )
    for (device, _), (
        (device_params_, device_grads_, device_momentum_buffer_list),
        _,
    ) in grouped_tensors.items():
        device_params: List[Tensor] = cast(List[Tensor], device_params_)
        device_grads: List[Tensor] = cast(List[Tensor], device_grads_)
        device_grad_scale, device_found_inf = None, None
        if grad_scale is not None:
            device_grad_scale = grad_scale_dict.setdefault(
                device, grad_scale.to(device)
            )
        if found_inf_dict is not None and found_inf is not None:
            device_found_inf = found_inf_dict.setdefault(device, found_inf.to(device))
        torch._fused_sgd_(
            device_params,
            device_grads,
            [],
            lr=lr,
            maximize=maximize,
            grad_scale=device_grad_scale,
            found_inf=device_found_inf,
        )
