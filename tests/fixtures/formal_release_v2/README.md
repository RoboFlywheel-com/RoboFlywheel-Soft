# Formal release fixture

`generate_formal_fixture.py` creates this synthetic, hashed v2/v12 fixture.
It contains verified, rejected, and task-failure physical rows so local tests
can prove that a 100% terminal execution population is not the same as a
100% release.  The fixture is never a production result and is deliberately
marked `synthetic=true`.
