"""Explicit component conversion, preserving checkpoint names and parameter identity."""


def adopt(module, implementation):
    if not issubclass(implementation, type(module)):
        raise TypeError('Component must extend the original implementation')
    if 'forward' in module.__dict__ or hasattr(module, '_hf_hook'):
        raise RuntimeError('Unexpected instance forward/offload hook; refusing implicit dispatch changes')
    result = implementation.__new__(implementation)
    result.__dict__ = module.__dict__.copy()
    # Keep all parameter/buffer objects and checkpoint names; own the dictionaries.
    for name in ('_parameters', '_buffers', '_modules'):
        result.__dict__[name] = module.__dict__[name].copy()
    if tuple(module.state_dict()) != tuple(result.state_dict()):
        raise RuntimeError('Component conversion changed checkpoint keys')
    return result


def replace_components(root, choose, inventory, prefix=''):
    for name, child in list(root.named_children()):
        full_name = prefix+'.'+name if prefix else name
        replace_components(child, choose, inventory, full_name)
        implementation = choose(full_name, child)
        if implementation is not None:
            converted = adopt(child, implementation)
            root.add_module(name, converted)
            inventory.append(dict(name=full_name, original=type(child).__name__, implementation=implementation.__name__))
