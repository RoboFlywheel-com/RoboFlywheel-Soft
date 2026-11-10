# Contributing

Use a virtual environment, install `requirements-dev.txt`, and run `python -m pytest -q` from the repository root. For analysis changes, verify the CSV hashes and regenerate the tables as described in `docs/REPRODUCIBILITY.md`.

Keep small synthetic fixtures in Git. Do not commit credentials, private cloud configuration, model checkpoints, production meshes or hidden evaluator targets. Read credentials from environment variables. Preserve revision identities, content hashes, terminal failure records and the distinction between synthetic tests and simulator evidence.

Describe the changed behavior, how you checked it, and any affected contract versions when submitting a change.

## Developer Certificate of Origin

All contributions must carry a `Signed-off-by:` line (`git commit -s`), certifying the contribution under the [Developer Certificate of Origin 1.1](https://developercertificate.org/):

> By making a contribution to this project, I certify that the contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or that it is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right to submit that work with modifications under the same license; and that I understand and agree that this project and the contribution are public and that a record of the contribution is maintained indefinitely.
