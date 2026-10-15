# utils.py
def requires_confirmation(func):
    func._requires_confirmation = True
    return func


def tool(description: str = "", category: str = "general"):
    def deco(func):
        func._tool_meta = {
            "description": description or (func.__doc__ or "").strip().split("\n")[0],
            "category": category,
            "name": func.__name__,
        }
        return func
    return deco