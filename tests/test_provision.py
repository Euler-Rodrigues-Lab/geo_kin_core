import pytest
from geo_kin_core.provision import _product_parts

def test_tianji_product():
    assert _product_parts('tianji-sharpa')==('tianji','sharpa')
    assert _product_parts('rby1-xhand')==('rby1','xhand')
    with pytest.raises(ValueError):_product_parts('not-a-robot')
