# Independent migration implementation decision

Decision: **approved for the exact 0293812 migration implementation**, with no remaining blocking source finding. The machine-readable receipt binds the adapter, deployment recipe and tests. Thirteen focused tests passed independently, including competing-process commits, busy-lock recovery, source identity checks, incomplete tails, cancellation/cap handling, history-to-pointer crash recovery and NFS visibility grace.

This approves source only. No migration, second-node proof, complete corpus or scientific training result is claimed. The approved native helper, validator and scientific evidence review remain unchanged. The earlier checkpoint findings are preserved as review history and are resolved by the reviewed implementation.

Runtime admission still requires fresh live-state checks, actual qualified work on all eight second-node GPUs before stopping old work, cross-node locking evidence, the durable migration boundary and complete source-bound ordinal/provenance validation. If the original generation has completed while authentication was unavailable, the stale migration must not be forced. The protected holder and holding successor remain outside cancellation scope.

Recovery and supervision are bounded. In particular, the reviewed deployment's two-day CPU limits are earlier than the seven-day in-process bounds. Cancellation, ordinary failure and OOM remain blocking conditions with preserved data. Existing native data QA and staged training gates retain authority; this receipt does not replace them. Parent reports that authentication timed out and no cluster deployment or job mutation occurred.
