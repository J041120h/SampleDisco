"""Pipeline wrappers (``wrapper``, ``rna_wrapper``, ``atac_wrapper``, ``multiomics_wrapper``).

This subpackage shares its name with the top-level entry point
``sampledisco.wrapper(...)``. Importing any submodule binds this *package* as the
attribute ``sampledisco.wrapper``, so the package itself is made callable: calling
it runs :func:`sampledisco.wrapper.wrapper.wrapper`. Both spellings therefore work
in any import order, and every dotted import path (``sampledisco.wrapper.wrapper``,
``sampledisco.wrapper.rna_wrapper``, ...) keeps working::

    from sampledisco import wrapper
    wrapper(**config)

Heavy dependencies are still imported only on the first call.
"""
import sys
import types


def _pipeline_wrapper():
    from sampledisco.wrapper.wrapper import wrapper
    return wrapper


class _CallableWrapperPackage(types.ModuleType):
    def __call__(self, *args, **kwargs):
        return _pipeline_wrapper()(*args, **kwargs)

    @property
    def __signature__(self):
        import inspect
        return inspect.signature(_pipeline_wrapper())


sys.modules[__name__].__class__ = _CallableWrapperPackage
