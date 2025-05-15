# utils/registry.py

METHOD_REGISTRY = {}

def register_method(name):
    """
    使用方式：@register_method("awq")
    """
    def decorator(func):
        if name in METHOD_REGISTRY:
            raise ValueError(f"Method '{name}' is already registered.")
        METHOD_REGISTRY[name] = func
        return func
    return decorator

def get_method(name):
    """
    根据方法名获取对应的运行函数。
    """
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Method '{name}' is not registered. Available methods: {list(METHOD_REGISTRY.keys())}")
    return METHOD_REGISTRY[name]

def list_methods():
    """
    返回所有注册的方法名（可用于 CLI 帮助打印）。
    """
    return list(METHOD_REGISTRY.keys())
