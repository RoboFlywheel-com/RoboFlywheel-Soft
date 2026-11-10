# Local fixtures

The executable fixture set is generated in memory by `softsim-physical-agent
smoke`; it contains first-pass, corrected, and budget-exhausted tasks.  This
directory intentionally contains no stage3 or cloud assets.  Tests create
small temporary legacy/stage3-shaped metadata trees and verify that artifact
references remain relative and content-addressed.

