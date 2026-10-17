import pytest
from geo_kin_core.provision import _product_parts

@pytest.mark.parametrize('product,expected', [
    ('dual_kinova3', ('dual_kinova3', None)),
    ('dual_kinova3-psyonic', ('dual_kinova3', 'psyonic')),
    ('rby1-xhand', ('rby1', 'xhand')),
])
def test_product_parts(product, expected):
    assert _product_parts(product) == expected


def test_free_product_install_does_not_require_active_license(monkeypatch,tmp_path,capsys):
    from types import SimpleNamespace
    from geo_kin_core import provision as p
    wheel=tmp_path/'free.whl';shared=tmp_path/'shared'
    monkeypatch.setattr(p,'_project_product',lambda *a:'dual_kinova3-psyonic')
    monkeypatch.setattr(p,'_project_python',lambda *a:tmp_path/'python')
    monkeypatch.setattr(p,'_registered_wheel',lambda *a:wheel)
    monkeypatch.setattr(p,'_shared_install',lambda *a:shared)
    monkeypatch.setattr(p,'_link_environment',lambda *a:tmp_path/'backend.pth')
    monkeypatch.setattr(p.subprocess,'check_output',lambda *a,**k:'["dual_kinova3-psyonic"]')
    monkeypatch.setattr(p,'_active_license',lambda:pytest.fail('free product must not resolve a license'))
    calls=[]
    monkeypatch.setattr(p.subprocess,'run',lambda *a,**k:calls.append((a,k)))
    p.install(SimpleNamespace(project=tmp_path,product=None,python=None,wheel=None))
    assert calls and calls[0][1]['check'] is True
    assert 'not required' in capsys.readouterr().out


def test_licensed_product_still_resolves_license(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from geo_kin_core import provision as p
    monkeypatch.setattr(p,'_project_product',lambda *a:'rby1-xhand')
    monkeypatch.setattr(p,'_project_python',lambda *a:tmp_path/'python')
    monkeypatch.setattr(p,'_registered_wheel',lambda *a:tmp_path/'paid.whl')
    monkeypatch.setattr(p,'_shared_install',lambda *a:tmp_path/'shared')
    monkeypatch.setattr(p,'_link_environment',lambda *a:tmp_path/'backend.pth')
    monkeypatch.setattr(p.subprocess,'check_output',lambda *a,**k:'["dual_kinova3-psyonic"]')
    def missing():raise FileNotFoundError('license required')
    monkeypatch.setattr(p,'_active_license',missing)
    with pytest.raises(FileNotFoundError,match='license required'):
        p.install(SimpleNamespace(project=tmp_path,product=None,python=None,wheel=None))


def test_failed_install_does_not_replace_current_wheel(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import subprocess
    from geo_kin_core import provision as p
    monkeypatch.setattr(p,'_project_product',lambda *a:'dual_kinova3')
    monkeypatch.setattr(p,'_project_python',lambda *a:tmp_path/'python')
    monkeypatch.setattr(p,'_registered_wheel',lambda *a:tmp_path/'bad.whl')
    monkeypatch.setattr(p,'_shared_install',lambda *a:tmp_path/'shared')
    monkeypatch.setattr(p.subprocess,'check_output',lambda *a,**k:'["dual_kinova3"]')
    monkeypatch.setattr(p,'_link_environment',lambda *a:pytest.fail('failed install must preserve dev selection'))
    def fail(*args,**kwargs):raise subprocess.CalledProcessError(1,args)
    monkeypatch.setattr(p.subprocess,'run',fail)
    with pytest.raises(subprocess.CalledProcessError):
        p.install(SimpleNamespace(project=tmp_path,product=None,python=None,wheel=None))


def test_explicit_python_preserves_virtualenv_symlink(tmp_path):
    import sys
    from geo_kin_core.provision import _project_python, _site_packages
    import subprocess
    env = tmp_path / 'venv'
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(env)], check=True)
    python = env / 'bin/python'
    selected = _project_python(tmp_path, str(python))
    assert selected == python
    assert _site_packages(selected).is_relative_to(env)
