"""Smoke-check an installed release, outside the source checkout."""
from importlib.metadata import version
assert version('eurola-geo-kin-core') == "0.1.1"
from geo_kin_core.provision import _product_parts
assert _product_parts('openarm') == ('openarm', None)
assert _product_parts('dual_kinova3-psyonic') == ('dual_kinova3', 'psyonic')
try:
    _product_parts('unknown')
except ValueError as error:
    assert not any(name in str(error) for name in ['g1', 'rby1', 'vega', 'openarm'])
else:
    raise AssertionError('invalid product accepted')
print("Release smoke check passed")
