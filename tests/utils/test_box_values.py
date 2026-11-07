import pytest

from mflux.utils.box_values import BoxValueError, BoxValues

pytestmark = pytest.mark.fast


@pytest.mark.parametrize("value", ["a%", "%", "%%", "1%2%", "%10%", "1.5%", " %"])
def test_parse_rejects_malformed_percent_with_box_value_error(value):
    # A string that merely ends with "%" is not a valid percentage. parse() must
    # reject it with BoxValueError instead of storing it and deferring an
    # uncaught ValueError to normalize_to_dimensions() at generation time.
    with pytest.raises(BoxValueError):
        BoxValues.parse(value)


@pytest.mark.parametrize(
    "value",
    ["10", "10%", "-5%", "10,20", "10%,20%", "1,2,3", "1,2,3,4", "10%,50,20%,50"],
)
def test_parse_then_normalize_never_raises_raw_exception(value):
    # Contract: whatever parse() accepts, normalize_to_dimensions() must handle
    # without raising a raw (non-BoxValueError) exception.
    box = BoxValues.parse(value)
    box.normalize_to_dimensions(512, 512)


def test_percent_is_resolved_against_the_matching_dimension():
    box = BoxValues.parse("10%,20%,10%,20%")
    absolute = box.normalize_to_dimensions(width=1000, height=500)
    # top/bottom scale with height, right/left with width.
    assert (absolute.top, absolute.bottom) == (50, 50)
    assert (absolute.right, absolute.left) == (200, 200)


def test_error_message_reports_the_offending_value():
    # Too many parts must surface the actual input, not a literal "{value}".
    with pytest.raises(BoxValueError) as excinfo:
        BoxValues.parse("1,2,3,4,5")
    message = str(excinfo.value)
    assert "{value}" not in message
    assert "1,2,3,4,5" in message
