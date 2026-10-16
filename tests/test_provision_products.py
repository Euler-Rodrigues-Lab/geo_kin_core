import pytest
from geo_kin_core.provision import _product_parts

@pytest.mark.parametrize('product,expected', [
    ('dual_kinova3', ('dual_kinova3', None)),
    ('dual_kinova3-psyonic', ('dual_kinova3', 'psyonic')),
    ('rby1-xhand', ('rby1', 'xhand')),
])
def test_product_parts(product, expected):
    assert _product_parts(product) == expected
