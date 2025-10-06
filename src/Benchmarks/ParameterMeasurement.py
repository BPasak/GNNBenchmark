from torch import nn


def countParameters(module: nn.Module) -> int:
    parameter_count: int = 0
    modules = list(module.modules())
    for i in range(1, len(modules)):
        if isinstance(modules[i], nn.Conv2d) or isinstance(modules[i], nn.Linear):
            parameter_count += modules[i].weight.nelement()
            if modules[i].bias is not None:
                parameter_count += modules[i].bias.nelement()
        elif isinstance(modules[i], nn.Module):
            parameter_count += countParameters(modules[i])

    return parameter_count