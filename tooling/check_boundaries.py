"""Check Python package ownership and declared workspace dependencies."""

import ast
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = {
    "omlx_config": "packages/config",
    "omlx_contracts": "packages/contracts",
    "omlx_management": "packages/management",
    "omlx_runtime": "packages/runtime",
    "omlx_server": "apps/server",
    "omlx_cli": "apps/cli",
}
ALLOWED = {
    "omlx_config": set(),
    "omlx_contracts": set(),
    "omlx_runtime": {"omlx_config", "omlx_contracts"},
    "omlx_management": {"omlx_config", "omlx_contracts", "omlx_runtime"},
    "omlx_server": {"omlx_config", "omlx_contracts", "omlx_runtime", "omlx_management"},
    "omlx_cli": {"omlx_config", "omlx_contracts", "omlx_runtime", "omlx_server"},
}


def violations(root: Path = ROOT):
    metadata = {
        ns: tomllib.loads((root / project / "pyproject.toml").read_text())["project"]
        for ns, project in PROJECTS.items()
    }
    owners_by_name = {project["name"]: ns for ns, project in metadata.items()}
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    result = []
    for ns, project in PROJECTS.items():
        declared = {
            re.match(r"[\w.-]+", dep).group(0).lower().replace("_", "-")
            for dep in metadata[ns]["dependencies"]
        }
        if metadata[ns]["version"] != version:
            result.append(f"{project}/pyproject.toml: version differs from workspace")
        for dependency in declared:
            owner = owners_by_name.get(dependency)
            if owner is not None and owner not in ALLOWED[ns]:
                result.append(
                    f"{project}/pyproject.toml: forbidden dependency {dependency}"
                )
        for path in (root / project / "src" / ns).rglob("*.py"):
            if "vendor" in path.parts or "_dashboard" in path.parts:
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    names = [node.module or ""]
                for name in names:
                    owner = name.split(".")[0]
                    if owner == ns or owner not in PROJECTS:
                        if ns == "omlx_runtime" and owner == "fastapi":
                            result.append(
                                f"{path.relative_to(root)}:{node.lineno}: runtime imports HTTP framework"
                            )
                        continue
                    if owner not in ALLOWED[ns]:
                        result.append(
                            f"{path.relative_to(root)}:{node.lineno}: {ns} cannot import {owner}"
                        )
                    elif metadata[owner]["name"] not in declared:
                        result.append(
                            f"{path.relative_to(root)}:{node.lineno}: undeclared dependency {metadata[owner]['name']}"
                        )
    return result


def main():
    errors = violations()
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Workspace import boundaries passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
