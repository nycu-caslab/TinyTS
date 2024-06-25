from MemPlannerAddon import *

def FindAddonByName(name:str) -> BaseAddon:
    if type(name) != str:
        dummy = ""
        raise(BaseException(f"Invalid arg type: expect '{type(dummy)}', receive '{type(name)}'."))
    result = AddonsTable.get(name, None)
    if result is not None:
        return result
    else:
        raise(BaseException(f"Unknown Addon Name: {name}"))

def AttachAddonsToPlan(reqs, ops, params, tensors, *addons):
    for addon_name in addons:
        addon_cls = FindAddonByName(addon_name)
        addon_cls.attach(reqs, ops, params, tensors)
