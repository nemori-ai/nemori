# Nemori SWE repository-memory experiment

This project exposes one SWE-bench experiment: source code from each task's
repository is ingested into native Nemori memory before the Coding Agent runs.
There are no Gold, Baseline, paired, or user-supplied snapshot modes.

## What is remembered

SWE-bench Lite contains tasks from multiple GitHub repositories. Each task
identifies a repository and `base_commit`. The pipeline starts from the
official SWE-bench image, resets the working tree to that commit, removes
remotes and future Git state, and exports only the working-tree files.

The repository scanner includes bounded source, test, dependency, build and
documentation files. It excludes `.git`, dependency/vendor trees, binaries,
generated outputs, caches, environment files and credentials. Issue-seeded
ranking limits native LLM memory construction to the most relevant files while
retaining a hash manifest of the complete safe text-file view.

Selected observations are sent through:

```text
NemoriMemory.add_messages()
    -> NemoriMemory.flush()
    -> Episode Memory + Semantic Memory
    -> NemoriMemory.search() / UnifiedSearch(HYBRID)
    -> read-only Agent memory context
```

There is intentionally no generated-experience, custom lexical, or hash-vector
fallback. If PostgreSQL, Qdrant, the LLM, embeddings, or native Nemori memory
construction fails, the memory stage fails and records the reason.

## One-click pipeline

Start the dashboard and open `http://127.0.0.1:8765/`. The Nemori SWE page
offers one full pipeline and the same stages as individual controls:

1. Prepare the revision-pinned SWE-bench Lite tasks.
2. Export each official repository at `base_commit`.
3. Build native Episode and Semantic Memory from repository code.
4. Retrieve issue-relevant memory and run pinned mini-swe-agent `2.4.5`.
5. Score the generated patch with the official `swebench==4.1.0` Harness.
6. Report resolved/failed counts, tokens, time, indexed files and memory hits.

The default is three tasks. Data, repository images, service startup, paths,
JSON conversion and artifacts are automatic. Results are stored in
`runs/nemori-swe/<run-id>/`.

## Safety boundary

The Agent and memory builder may see only the normal Agent projection and the
repository working tree at `base_commit`. They never receive dataset `patch`,
`test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`, future commits, or Harness-only
records. Protected evaluator rows remain in a separate directory and are used
only after the candidate patch has been frozen.

This experiment tests whether Nemori-indexed repository code helps an Agent
complete software-engineering tasks. It is not reported as an official
SWE-bench leaderboard score.

## Runtime

- Docker Desktop with Linux containers
- Ubuntu WSL2 with `uv` at `$HOME/.local/bin/uv`
- PostgreSQL and Qdrant (started automatically through Docker Compose)
- configured Agent/Nemori LLM and embedding credentials

The dashboard does not require manual dataset downloads, repository checkouts,
prediction files, memory files, or filesystem paths.
