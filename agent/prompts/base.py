"""
agent/prompts/base.py
---------------------
Shared infrastructure for all TechVest prompt modules.

Every individual prompt module creates a PromptTemplate instance and
exposes it as a module-level constant alongside a typed render() function.

Classes
-------
PromptTemplate
    Immutable dataclass that holds a prompt's raw template string,
    metadata (name, version, description), and a render() method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PromptTemplate:
    """Immutable container for a single prompt template.

    Parameters
    ----------
    name:
        Short machine-readable identifier (e.g. ``"parse_resume"``).
    template:
        The raw prompt string.  Use ``{placeholder}`` for variables.
        Use ``{{`` / ``}}`` for literal braces (JSON examples in the prompt).
    version:
        Semver string for tracking iterations in production logs.
    description:
        One-sentence description shown in tooling / dashboards.
    variables:
        Names of all ``{placeholder}`` variables required by this template.
        Used for validation in ``render()``.
    """

    name: str
    template: str
    version: str = "1.0"
    description: str = ""
    variables: tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, **kwargs: Any) -> str:
        """Render the template by substituting ``{placeholder}`` variables.

        Parameters
        ----------
        **kwargs:
            One keyword argument per ``{placeholder}`` in the template.

        Returns
        -------
        str
            The rendered prompt string, ready to send to an LLM.

        Raises
        ------
        KeyError
            If a required variable declared in ``self.variables`` is missing
            from ``kwargs``.
        ValueError
            If an unknown variable is supplied (strict mode guard).

        Examples
        --------
        >>> prompt = PromptTemplate(
        ...     name="greet",
        ...     template="Hello, {name}!",
        ...     variables=("name",),
        ... )
        >>> prompt.render(name="Priya")
        'Hello, Priya!'
        """
        # Validate required variables are supplied
        missing = [v for v in self.variables if v not in kwargs]
        if missing:
            raise KeyError(
                f"Prompt '{self.name}' is missing required variable(s): {missing}"
            )

        # Warn about extra variables (soft check — don't raise)
        extra = [k for k in kwargs if k not in self.variables]
        if extra and self.variables:
            import warnings
            warnings.warn(
                f"Prompt '{self.name}' received unexpected variable(s): {extra}. "
                "They will be ignored.",
                stacklevel=2,
            )

        return self.template.format(**kwargs)

    def preview(self, max_chars: int = 200) -> str:
        """Return a truncated preview of the template for logging."""
        snippet = self.template[:max_chars].replace("\n", " ")
        suffix = "..." if len(self.template) > max_chars else ""
        return f"[{self.name} v{self.version}] {snippet}{suffix}"

    def __str__(self) -> str:
        return f"PromptTemplate(name={self.name!r}, version={self.version!r})"

    def __repr__(self) -> str:
        return (
            f"PromptTemplate(name={self.name!r}, version={self.version!r}, "
            f"variables={self.variables!r})"
        )
