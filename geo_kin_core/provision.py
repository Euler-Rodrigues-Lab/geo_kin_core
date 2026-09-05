"""Manage licensed geo_kin wheels without storing them in consumer repos."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


def config_root() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "geo-kin"


def data_root() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "geo-kin"


def license_dir() -> Path:
    return config_root() / "licenses"


def wheel_dir() -> Path:
    override = os.environ.get("GEO_KIN_WHEEL_DIR")
    return Path(override).expanduser() if override else data_root() / "wheels"


def install_dir() -> Path:
    return data_root() / "installs"


def active_license_path() -> Path:
    return config_root() / "license.toml"


def _read_toml(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _private_file(path: Path) -> None:
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _copy_checked(source: Path, destination: Path, *, replace: bool = False) -> bool:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(source) == _sha256(destination):
            return False
        if not replace:
            raise FileExistsError(
                f"{destination} already exists with different contents; pass --replace"
            )
    shutil.copy2(source, destination)
    return True


def _license_name(path: Path, document: dict, requested: str | None) -> str:
    if requested:
        name = requested
    else:
        licensee = str(document.get("licensee", path.stem))
        name = "".join(char.lower() if char.isalnum() else "-" for char in licensee)
        name = "-".join(filter(None, name.split("-")))
    if not name or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for char in name):
        raise ValueError("license name may contain only lowercase letters, digits, '-' and '_'")
    return name


def _product_parts(product: str) -> tuple[str, str | None]:
    parts = product.lower().split("-", 1)
    robot = parts[0]
    hand = parts[1] if len(parts) == 2 else None
    if robot not in {"g1", "rby1", "vega"}:
        raise ValueError("product must begin with g1, rby1, or vega")
    return robot, hand


def _project_product(project: Path, explicit: str | None) -> str:
    if explicit:
        _product_parts(explicit)
        return explicit.lower()
    pyproject = project / "pyproject.toml"
    if not pyproject.is_file():
        raise FileNotFoundError(f"no pyproject.toml under {project}")
    product = _read_toml(pyproject).get("tool", {}).get("geo-kin", {}).get("product")
    if not product:
        raise ValueError(
            f"{pyproject} has no [tool.geo-kin] product; pass --product explicitly"
        )
    _product_parts(str(product))
    return str(product).lower()


def _project_python(project: Path, explicit: str | None) -> Path:
    if explicit:
        python = Path(explicit).expanduser().resolve()
    elif os.environ.get("VIRTUAL_ENV") and Path.cwd().resolve() == project:
        executable = "Scripts/python.exe" if os.name == "nt" else "bin/python"
        python = Path(os.environ["VIRTUAL_ENV"]) / executable
    else:
        python = project / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        raise FileNotFoundError(f"Python environment not found at {python}; run `uv sync` first")
    return python


def _registered_wheel(product: str, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = sorted((wheel_dir() / product).glob("geo_kin-*.whl"))
    if not candidates:
        raise FileNotFoundError(
            f"no wheel registered for {product!r} under {wheel_dir() / product}; "
            "run `geo-kin-provision register` first"
        )
    return candidates[-1]


def _active_license() -> Path:
    explicit = os.environ.get("GEO_KIN_LICENSE")
    path = Path(explicit).expanduser() if explicit else active_license_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"no active license at {path}; register one or run `geo-kin-provision activate NAME`"
        )
    return path.resolve()


def _shared_install(wheel: Path, product: str) -> Path:
    digest = _sha256(wheel)
    destination = install_dir() / product / f"{wheel.stem}-{digest[:12]}"
    marker = destination / ".geo-kin-wheel-sha256"
    if marker.is_file() and marker.read_text().strip() == digest:
        return destination

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"unsafe path in wheel: {member.filename}")
        archive.extractall(destination)
    marker.write_text(f"{digest}\n")
    return destination


def _site_packages(python: Path) -> Path:
    command = [
        str(python),
        "-c",
        "import sysconfig; print(sysconfig.get_paths()['purelib'])",
    ]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    path = Path(output.strip()).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"site-packages not found at {path}")
    return path


def _link_environment(python: Path, shared: Path) -> Path:
    link = _site_packages(python) / "geo_kin_shared.pth"
    # Prepend so a stale, directly installed copy cannot shadow the selected
    # centrally managed build. uv leaves unmanaged .pth files intact on sync.
    link.write_text(f"import sys; sys.path.insert(0, {str(shared)!r})\n")
    return link


def register(args) -> None:
    product = args.product.lower()
    _product_parts(product)
    changed = []
    if args.wheel:
        source = Path(args.wheel)
        if source.suffix != ".whl" or not source.name.startswith("geo_kin-"):
            raise ValueError("--wheel must be a geo_kin-*.whl file")
        destination = wheel_dir() / product / source.name
        if _copy_checked(source, destination, replace=args.replace):
            _private_file(destination)
            changed.append(f"wheel: {destination}")
        else:
            _private_file(destination)
            changed.append(f"wheel already registered: {destination}")

    registered_license = None
    if args.license:
        source = Path(args.license).expanduser().resolve()
        document = _read_toml(source)
        if document.get("schema") != "geo_kin.license/1":
            raise ValueError(f"{source} is not a geo_kin.license/1 document")
        required = {part for part in _product_parts(product) if part}
        features = set(document.get("features", []))
        if not required.issubset(features):
            raise ValueError(
                f"license features {sorted(features)} do not cover product {product!r}"
            )
        name = _license_name(source, document, args.name)
        registered_license = license_dir() / f"{name}.toml"
        if _copy_checked(source, registered_license, replace=args.replace):
            changed.append(f"license: {registered_license}")
        else:
            changed.append(f"license already registered: {registered_license}")
        _private_file(registered_license)

    if not args.wheel and not args.license:
        raise ValueError("register requires --wheel, --license, or both")

    active = active_license_path()
    if registered_license is not None and (args.activate or not active.exists()):
        _copy_checked(registered_license, active, replace=args.replace or args.activate)
        _private_file(active)
        changed.append(f"active license: {active}")
    elif registered_license is not None and active.exists():
        _private_file(active)
        changed.append(f"active license unchanged: {active}")
    print("\n".join(changed))


def activate(args) -> None:
    source = license_dir() / f"{args.name}.toml"
    if not source.is_file():
        available = ", ".join(path.stem for path in sorted(license_dir().glob("*.toml")))
        raise FileNotFoundError(f"unknown license {args.name!r}; available: {available or 'none'}")
    destination = active_license_path()
    _copy_checked(source, destination, replace=True)
    _private_file(destination)
    print(f"active license: {destination}")


def install(args) -> None:
    project = Path(args.project).expanduser().resolve()
    product = _project_product(project, args.product)
    robot, hand = _product_parts(product)
    python = _project_python(project, args.python)
    wheel = _registered_wheel(product, args.wheel)
    license_path = _active_license()
    shared = _shared_install(wheel, product)
    link = _link_environment(python, shared)

    check = """
import geo_kin
from pathlib import Path
assert Path(geo_kin.__file__).resolve().is_relative_to(Path(SHARED).resolve()), geo_kin.__file__
robots = set(geo_kin.COMPILED_ROBOTS)
hands = set(geo_kin.COMPILED_HANDS)
assert ROBOT in robots, (ROBOT, sorted(robots))
assert HAND is None or HAND in hands, (HAND, sorted(hands))
geo_kin.RetargetSession(robot=ROBOT, hand=HAND)
print(f"geo_kin {geo_kin.__version__}: {PRODUCT} session OK")
"""
    environment = os.environ.copy()
    environment["GEO_KIN_LICENSE"] = str(license_path)
    subprocess.run(
        [
            str(python),
            "-c",
            (
                f"ROBOT={robot!r}; HAND={hand!r}; PRODUCT={product!r}; "
                f"SHARED={str(shared)!r};\n{check}"
            ),
        ],
        check=True,
        env=environment,
    )
    print(f"project: {project}")
    print(f"python: {python}")
    print(f"wheel: {wheel}")
    print(f"shared install: {shared}")
    print(f"environment link: {link}")
    print(f"license: {license_path}")


def status(_args) -> None:
    print(f"wheel root: {wheel_dir()}")
    wheels = sorted(wheel_dir().glob("*/*.whl"))
    if wheels:
        for wheel in wheels:
            print(f"  {wheel.parent.name}: {wheel.name}")
    else:
        print("  no registered wheels")
    print(f"license root: {license_dir()}")
    licenses = sorted(license_dir().glob("*.toml"))
    if licenses:
        for path in licenses:
            document = _read_toml(path)
            print(
                f"  {path.stem}: features={document.get('features', [])} "
                f"expiry={document.get('expiry', 'unknown')}"
            )
    else:
        print("  no registered licenses")
    active = os.environ.get("GEO_KIN_LICENSE") or str(active_license_path())
    print(f"active license: {active}{'' if Path(active).expanduser().is_file() else ' (missing)'}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Register and install licensed geo_kin wheels from user-wide storage."
    )
    commands = result.add_subparsers(dest="command", required=True)

    register_parser = commands.add_parser("register", help="store a supplied wheel/license")
    register_parser.add_argument("--product", required=True, help="for example rby1-xhand")
    register_parser.add_argument("--wheel")
    register_parser.add_argument("--license")
    register_parser.add_argument("--name", help="short stored license name")
    register_parser.add_argument("--activate", action="store_true")
    register_parser.add_argument("--replace", action="store_true")
    register_parser.set_defaults(handler=register)

    activate_parser = commands.add_parser("activate", help="select a registered license")
    activate_parser.add_argument("name")
    activate_parser.set_defaults(handler=activate)

    install_parser = commands.add_parser("install", help="install and verify a consumer venv")
    install_parser.add_argument("--project", default=".")
    install_parser.add_argument("--product")
    install_parser.add_argument("--python", help="target Python executable")
    install_parser.add_argument("--wheel", help="one-off wheel path instead of registered wheel")
    install_parser.set_defaults(handler=install)

    status_parser = commands.add_parser("status", help="show registered wheels and licenses")
    status_parser.set_defaults(handler=status)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        args.handler(args)
    except (FileExistsError, FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"geo-kin-provision: {exc}") from exc


if __name__ == "__main__":
    main()
