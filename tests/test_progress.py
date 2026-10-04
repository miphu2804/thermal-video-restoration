from thermal_restore.progress import format_duration


def test_format_duration():
    assert format_duration(3661.2) == "01:01:01"


def test_format_duration_clamps_negative_values():
    assert format_duration(-5) == "00:00:00"
