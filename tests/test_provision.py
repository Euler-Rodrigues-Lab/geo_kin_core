import pytest
from geo_kin_core.provision import _product_parts

@pytest.mark.parametrize('product,expected', [
    ('openarm', ('openarm', None)),
    ('g1-inspire', ('g1', 'inspire')),
    ('dual_kinova3-psyonic', ('dual_kinova3', 'psyonic')),
])
def test_release_products(product, expected):
    assert _product_parts(product) == expected


def test_invalid_product_does_not_enumerate_products():
    with pytest.raises(ValueError) as error:
        _product_parts('unknown')
    assert not any(name in str(error.value) for name in ('g1', 'rby1', 'vega', 'openarm', 'dual_kinova3'))
