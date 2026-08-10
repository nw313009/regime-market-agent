# sql/

**Intentionally empty. `setup/` is the single source of DDL.**

v1.0 planned this directory as a reviewable mirror of `setup/`: the same table definitions, without
the surrounding execution code. That plan is dropped as of B-6.

The reason is that a mirror is a SECOND definition of every table, hand-copied, with nothing testing
that the two agree. It would drift — and a drifted DDL reference is worse than no reference,
because it is the file a reader trusts while the cluster runs the other one. The A-2 and A-3
checkpoints already shipped their tables without updating a mirror, which is the drift arriving
before the copies even existed.

So there is exactly one place a table is defined:

| File | Contents |
| --- | --- |
| `setup/create_catalog.sql` | catalog + bronze/silver/gold schemas |
| `setup/create_delta_tables.sql` | every Delta table, with a COMMENT on every column |
| `setup/create_lakebase.sql` | the Lakebase (Postgres) tables, at checkpoint C |

`setup/create_delta_tables.sql` is written to be read as well as run: every column carries a
COMMENT, every statement is idempotent, and each layer has a header explaining what belongs in it.
Reviewing it is reviewing the schema. The tests in `tests/test_idempotency.py` and
`tests/test_features.py` parse that file directly and assert the declared columns, the NOT NULL
MERGE keys and the ledger vocabulary match the code — which is the check a mirror could never have.

The architectural requirement (architecture doc section 15) is that the environment can be
recreated from code rather than from manually configured workspace state. `setup/` satisfies it on
its own: run `create_catalog.sql`, then `create_delta_tables.sql`.

This file is the only thing in this directory, and it stays so that the decision is recorded where
someone would otherwise go looking for the mirror. Do not add DDL here.
