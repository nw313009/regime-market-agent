# sql/

Canonical DDL, kept in sync with `setup/` (spec B0).

`setup/` holds the executable bootstrap assets — the scripts you actually run to recreate the
environment. This directory holds the same DDL as reviewable, version-controlled reference:
the shape of every table in one place, without the surrounding execution code.

The architectural requirement (architecture doc section 15) is that the environment can be
recreated from code rather than from manually configured workspace state. So the rule is: if
a table definition changes in `setup/`, it changes here in the same commit.

Expected contents as the checkpoints land:

| File | Mirrors | Checkpoint |
| --- | --- | --- |
| `create_catalog.sql` | `setup/create_catalog.sql` | A |
| `create_delta_tables.sql` | `setup/create_delta_tables.sql` | A, B |
| `create_lakebase.sql` | `setup/create_lakebase.sql` | C |

This file is a placeholder so the directory exists in git; git does not track empty
directories.
