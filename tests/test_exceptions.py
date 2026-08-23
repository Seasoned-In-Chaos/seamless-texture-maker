import pytest

from app.core.exceptions import (
    SeamsError,
    ImageLoadError,
    ProcessingError,
    GPUError,
    CacheError,
)


@pytest.mark.parametrize(
    "exc_cls", [ImageLoadError, ProcessingError, GPUError, CacheError]
)
def test_subclasses_seams_error(exc_cls):
    assert issubclass(exc_cls, SeamsError)


@pytest.mark.parametrize(
    "exc_cls", [SeamsError, ImageLoadError, ProcessingError, GPUError, CacheError]
)
def test_subclasses_exception(exc_cls):
    assert issubclass(exc_cls, Exception)


def test_carries_message():
    with pytest.raises(ImageLoadError, match="bad file"):
        raise ImageLoadError("bad file")


def test_catchable_as_base_class():
    try:
        raise ProcessingError("failed")
    except SeamsError as exc:
        assert str(exc) == "failed"
    else:
        pytest.fail("should have been caught as SeamsError")
