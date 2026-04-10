"""
Legacy notebook-oriented analysis helpers.

This module intentionally stays minimal because the ADMDynAnlz software does not
import or rely on the old broad `adam_functions.py` utility collection. The
previous catch-all helper module contained a large amount of dead code and many
heavy optional dependencies that were not part of the active application.

If notebook-only utilities are needed later, add them here deliberately and keep
them scoped to a specific use case.
"""

__all__: list[str] = []
