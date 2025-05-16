# utils/registry.py

METHOD_REGISTRY = {}

def register_method(name):

    def decorator(func):
        if name in METHOD_REGISTRY:
            raise ValueError(f"Method '{name}' is already registered.")
        METHOD_REGISTRY[name] = func
        return func
    return decorator

def get_method(name):

    if name not in METHOD_REGISTRY:
        raise ValueError(f"Method '{name}' is not registered. Available methods: {list(METHOD_REGISTRY.keys())}")
    return METHOD_REGISTRY[name]

def list_methods():

    return list(METHOD_REGISTRY.keys())
