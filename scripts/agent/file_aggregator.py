import ast
import importlib.util
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

from rfstudio.engine.task import Task
from rfstudio.ui import console


@dataclass
class FileAggregator(Task):

    """
    Collects a script and all its local Python dependencies into a single folder,
    maintaining the directory structure so the script can be run from there.
    """

    input: Path = ...
    """The path to the input Python script."""

    output: Path = ...
    """The path to the output directory."""

    ignore: List[str] = field(default_factory=list)
    """
    Prefixes of modules to ignore and not copy (e.g., ['torch', 'numpy']).
    """

    def run(self) -> None:
        if not self.input.exists() or not self.input.is_file():
            console.print(f"[bold red]Error:[/bold red] Input must be an existing file. Got: {self.input}")
            return

        project_root = self._find_project_root(self.input)
        if project_root is None:
            console.print(
                "[bold red]Error:[/bold red] Could not determine project root (marker: pyproject.toml). "
                "Using input file's directory as fallback."
            )
            project_root = self.input.resolve().parent

        console.print(f"Project root identified: {project_root}")
        console.print(f"Output directory: {self.output}")

        if self.output.exists():
            shutil.rmtree(self.output)
        self.output.mkdir(parents=True)

        files_to_process: List[Path] = [self.input.resolve()]
        processed_files: Set[Path] = set()

        with console.status('Aggregating files...'):
            while files_to_process:
                current_file = files_to_process.pop(0)
                if current_file in processed_files:
                    continue

                try:
                    relative_path = current_file.relative_to(project_root)
                except ValueError:
                    console.print(f"[yellow]Skipping file outside project root:[/yellow] {current_file}")
                    continue

                dest_path = self.output / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(current_file, dest_path)
                processed_files.add(current_file)

                for parent in relative_path.parents:
                    if parent == Path('.'):
                        continue
                    init_py = project_root / parent / '__init__.py'
                    if init_py.exists() and init_py not in processed_files:
                        files_to_process.append(init_py)

                if current_file.suffix == '.py':
                    try:
                        with open(current_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        tree = ast.parse(content, filename=str(current_file))

                        imported_files = self._find_imports(tree, current_file, project_root)
                        files_to_process.extend(imported_files)
                    except Exception as e:
                        console.print(f"[bold red]Error parsing {current_file}:[/bold red] {e}")
        console.print(f"[green]Successfully aggregated {len(processed_files)} files to {self.output}[/green]")


    def _find_project_root(self, start_path: Path) -> Optional[Path]:
        current_dir = start_path.resolve().parent
        while current_dir != current_dir.parent:
            if (current_dir / 'pyproject.toml').exists():
                return current_dir
            current_dir = current_dir.parent
        return None

    def _find_imports(self, tree: ast.AST, file_path: Path, project_root: Path) -> List[Path]:
        imported_files = []

        try:
            package = ".".join(file_path.parent.relative_to(project_root).parts)
        except ValueError:
            return []

        for node in ast.walk(tree):
            module_name_to_resolve = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name_to_resolve = alias.name
                    module_path = self._resolve_module(module_name_to_resolve, project_root)
                    if module_path:
                        imported_files.append(module_path)

            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    relative_module_name = "." * node.level + (node.module or "")
                    try:
                        module_name_to_resolve = importlib.util.resolve_name(relative_module_name, package)
                    except (ImportError, ValueError):
                        continue
                elif node.module:
                    module_name_to_resolve = node.module

                if module_name_to_resolve:
                    module_path = self._resolve_module(module_name_to_resolve, project_root)
                    if module_path:
                        imported_files.append(module_path)

        return imported_files

    def _get_site_packages_paths(self) -> List[Path]:
        if not hasattr(self, '_site_packages_paths_cache'):
            import sysconfig
            paths = sysconfig.get_paths()
            self._site_packages_paths_cache = [Path(p) for k, p in paths.items() if k in ('purelib', 'platlib')]
        return self._site_packages_paths_cache

    def _resolve_module(self, module_name: str, project_root: Path) -> Optional[Path]:
        if self.ignore and any(module_name.startswith(p) for p in self.ignore):
            return None

        module_parts = module_name.split('.')

        potential_file = project_root.joinpath(*module_parts).with_suffix('.py')
        if potential_file.is_file():
            return potential_file.resolve()

        potential_package_init = project_root.joinpath(*module_parts, '__init__.py')
        if potential_package_init.is_file():
            return potential_package_init.resolve()

        try:
            spec = importlib.util.find_spec(module_name)
            if spec and spec.origin and spec.origin not in ('built-in', 'frozen'):
                module_path = Path(spec.origin).resolve()

                site_packages_paths = self._get_site_packages_paths()
                is_in_site_packages = any(
                    site_path in module_path.parents or site_path == module_path.parent
                    for site_path in site_packages_paths
                )
                if is_in_site_packages:
                    return None

                if project_root in module_path.parents or project_root == module_path.parent:
                    return module_path
        except (ModuleNotFoundError, ValueError):
            pass

        return None

if __name__ == '__main__':
    FileAggregator().run()
