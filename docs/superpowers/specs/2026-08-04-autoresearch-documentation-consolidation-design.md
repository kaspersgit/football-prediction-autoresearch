# Autoresearch documentation consolidation design

## Goal

Remove the competing sources of truth for experiments, evaluation rules, and runtime defaults without losing historical experiment evidence.

## Structure

The repository will use five documents with distinct responsibilities:

- `README.md`: installation, supported commands, and current user-facing behaviour.
- `autoresearch/GUIDE.md`: the agent's iteration procedure and file boundaries.
- `autoresearch/EVALUATION.md`: stable evaluation rules and statistical interpretation.
- `autoresearch/current.md`: the latest verified configuration, metrics, and active experiment queue.
- `autoresearch/experiments.md`: the append-only experiment record.

`docs/index.html` remains the GitHub Pages landing page. It is not part of the research record.

## History migration

`autoresearch/state.md` and `docs/improvements.md` currently form two independent experiment histories. The consolidated ledger will preserve both histories and label their origin. Existing iteration numbers are retained as legacy identifiers, while globally unique IDs based on the experiment date and source prevent collisions.

The migration must retain findings that only exist in `docs/improvements.md`, including its pending xG and edge-baseline investigations. Once migrated, the two old ledgers are removed to prevent further writes to them.

## Runtime configuration

Backtesting and prediction currently repeat the same betting defaults. A focused configuration module will define the shared defaults for maximum odds, maximum edge, maximum overround, and excluded betting leagues. Both execution paths will import these values.

The research threshold remains an explicit command-line choice (`--threshold 0.0`); the normal CLI default remains `0.03`, and `predict.sh` retains its production default of `0.04`.

## Documentation rules

- Dated metrics belong in `current.md` or the experiment ledger, not in the evergreen evaluation policy.
- The active queue contains only untested work.
- README claims must match executable behaviour.
- The guide must explicitly run the research configuration and require the evaluation policy to be read.
- Runtime defaults should be stated once in code and covered by tests.

## Verification

- Regression tests confirm that shared defaults are used by both backtesting and prediction.
- The full test suite must pass.
- A repository search must find no active links to the removed ledgers and no conflicting legacy claims in the README or guide.
- The consolidated ledger must contain content from both original histories and document the legacy-ID collision.
