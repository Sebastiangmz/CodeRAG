"""Lightweight code graph extraction for supported source languages."""

from __future__ import annotations

import re

from coderag.models.document import Document
from coderag.models.graph import CodeEdge, CodeSymbol, FileGraph

_PY_SYMBOL_RE = re.compile(r"^(?P<indent>\s*)(?P<kind>class|def|async\s+def)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)")
_PY_IMPORT_RE = re.compile(r"^(?:from\s+[A-Za-z0-9_.]+\s+import\s+(?P<from>.+)|import\s+(?P<import>.+))")
_TS_CLASS_RE = re.compile(r"^(?:export\s+)?class\s+(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)")
_TS_FUNCTION_RE = re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)")
_TS_METHOD_RE = re.compile(r"^(?:async\s+)?(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*\([^)]*\)\s*[:A-Za-z0-9_<>,\s|]*\{")
_TS_IMPORT_RE = re.compile(r"^import\s+(?:\{(?P<named>[^}]+)\}|(?P<default>[A-Za-z_$][A-Za-z0-9_$]*))")
_CALL_RE = re.compile(r"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*\(")
_KEYWORDS = {"if", "for", "while", "switch", "return", "function", "def", "class", "new"}
_SUPPORTED = {"python", "typescript", "javascript"}


class CodeGraphExtractor:
    """Extract symbols and simple import/call edges from one document."""

    def extract(self, document: Document) -> FileGraph:
        language = document.language
        if language not in _SUPPORTED:
            return FileGraph(
                repo_id=document.repo_id,
                file_path=document.file_path,
                language=language,
                capabilities={"graph_supported": False, "fallback": "text"},
            )
        if language == "python":
            symbols, edges = self._extract_python(document)
        else:
            symbols, edges = self._extract_typescript(document)
        return FileGraph(
            repo_id=document.repo_id,
            file_path=document.file_path,
            language=language,
            symbols=symbols,
            edges=edges,
            capabilities={"graph_supported": True, "languages": sorted(_SUPPORTED)},
        )

    def _extract_python(self, document: Document) -> tuple[list[CodeSymbol], list[CodeEdge]]:
        symbols: list[CodeSymbol] = []
        edges: list[CodeEdge] = []
        container_stack: list[tuple[int, str]] = []
        for line_number, line in enumerate(document.content.splitlines(), 1):
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            container_stack = [(level, name) for level, name in container_stack if level < indent]
            match = _PY_SYMBOL_RE.match(line)
            if match:
                raw_kind = match.group("kind")
                kind = "class" if raw_kind == "class" else "function"
                if kind == "function" and container_stack:
                    kind = "method"
                name = match.group("name")
                symbols.append(
                    CodeSymbol(
                        repo_id=document.repo_id,
                        file_path=document.file_path,
                        name=name,
                        kind=kind,
                        language=document.language,
                        start_line=line_number,
                        end_line=line_number,
                        container=container_stack[-1][1] if container_stack else None,
                    )
                )
                container_stack.append((indent, name))
                continue
            import_match = _PY_IMPORT_RE.match(stripped)
            if import_match:
                targets = import_match.group("from") or import_match.group("import") or ""
                for target in _split_import_targets(targets):
                    edges.append(self._edge(document, "import", target, line_number, container_stack[-1][1] if container_stack else None))
            for call in _CALL_RE.finditer(stripped):
                target = call.group("name")
                if target not in _KEYWORDS:
                    edges.append(self._edge(document, "call", target, line_number, container_stack[-1][1] if container_stack else None))
        return symbols, edges

    def _extract_typescript(self, document: Document) -> tuple[list[CodeSymbol], list[CodeEdge]]:
        symbols: list[CodeSymbol] = []
        edges: list[CodeEdge] = []
        current_class: str | None = None
        class_depth = 0
        for line_number, line in enumerate(document.content.splitlines(), 1):
            stripped = line.strip()
            class_match = _TS_CLASS_RE.match(stripped)
            function_match = _TS_FUNCTION_RE.match(stripped)
            method_match = _TS_METHOD_RE.match(stripped) if current_class else None
            if class_match:
                current_class = class_match.group("name")
                class_depth = 1
                symbols.append(self._symbol(document, current_class, "class", line_number))
            elif function_match:
                symbols.append(self._symbol(document, function_match.group("name"), "function", line_number))
            elif method_match:
                symbols.append(self._symbol(document, method_match.group("name"), "method", line_number, current_class))
            import_match = _TS_IMPORT_RE.match(stripped)
            if import_match:
                targets = import_match.group("named") or import_match.group("default") or ""
                for target in _split_import_targets(targets):
                    edges.append(self._edge(document, "import", target, line_number, current_class))
            for call in _CALL_RE.finditer(stripped):
                target = call.group("name")
                if target not in _KEYWORDS:
                    edges.append(self._edge(document, "call", target, line_number, current_class))
            if current_class:
                class_depth += stripped.count("{") - stripped.count("}")
                if class_depth <= 0:
                    current_class = None
        return symbols, edges

    def _symbol(self, document: Document, name: str, kind: str, line_number: int, container: str | None = None) -> CodeSymbol:
        return CodeSymbol(document.repo_id, document.file_path, name, kind, document.language, line_number, line_number, container)

    def _edge(self, document: Document, edge_type: str, target: str, line_number: int, source: str | None) -> CodeEdge:
        return CodeEdge(document.repo_id, document.file_path, edge_type, target, line_number, line_number, source)


def _split_import_targets(raw: str) -> list[str]:
    targets: list[str] = []
    for part in raw.split(","):
        name = part.strip().split(" as ", 1)[0].strip()
        if not name:
            continue
        targets.append(name.split(".")[-1])
    return targets
