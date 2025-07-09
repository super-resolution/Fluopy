import inspect

from fluopy import fluo_data as fd


def test_subclasses():
    classes = inspect.getmembers(object=fd, predicate=inspect.isclass)
    for name, subclass in classes:
        if name != "FluorophoreData":
            assert issubclass(subclass, fd.FluorophoreData)
