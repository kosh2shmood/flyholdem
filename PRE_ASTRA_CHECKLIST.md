# FlyHoldem pre-Astra checklist

You do not need to download the connectome, DOOMFLY, PokerKit, PettingZoo, OpenSpiel, or a poker solver manually. Give Astra network permission and make verified downloads part of the implementation. This prevents the project from beginning with untracked or stale data.

## What is already available on this Mac

Checked on 2026-09-12:

- Apple Silicon Mac with 16 GB RAM;
- about 77 GB free disk space;
- Git 2.52;
- Apple Clang 16 and GNU Make;
- Node.js 24;
- Codex CLI 0.153.4 through the ChatGPT desktop application;
- Homebrew;
- system Python 3.14;
- `uv` and CMake are missing.

This is enough for repository work, the web dashboard, fixture/circuit experiments, and likely one carefully measured full-graph process. Full-connectome training will be slower and tighter on memory than on a 32 GB machine. Keep only one full-graph worker active, use compressed logs, and watch free disk space.

## Install before starting the Astra task

Install `uv` and the native build tools. Do not replace or relink the system Python:

```bash
brew install uv cmake pkg-config
```

Inside the FlyHoldem repository, let `uv` install its own managed Python 3.11 and create a project-only virtual environment:

```bash
uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate
python --version
```

This adds Python 3.11 alongside the existing Python 3.14. It does not modify `/usr/bin/python3`, replace Homebrew's default Python, or affect other projects. Leaving the environment with `deactivate` restores the shell's normal Python selection.

Confirm the remaining tools:

```bash
git --version
uv --version
node --version
clang --version
cmake --version
codex --version
```

Do not install Python or Node project dependencies globally. Astra should maintain `.venv`, `uv.lock`, and the Node lockfile inside the repository workflow. The `.venv` directory itself should be ignored by Git.

## Prepare the workspace

1. Create or choose one persistent local project folder with at least 40 GB free. Avoid a temporary directory.
2. Initialize it as a Git repository and create the first commit containing the three planning documents.
3. Open that saved project directly in Codex. For the initial large-data build, prefer a local checkout over a disposable worktree so downloaded data and long-running checkpoints stay in one place.
4. Confirm that GPT-6 Astra is available in the model selector and choose high or xhigh reasoning.
5. Allow write access to the project folder and network access for GitHub, Python/Node package registries, and the official MaleCNS download host.
6. Allow a local server to bind to `localhost` for the dashboard when Codex requests it.
7. Keep the Mac connected to power and prevent sleep during full training runs.

The project should initially contain:

```text
flyholdem/
  flyholdem_build_spec.md
  ASTRA_IMPLEMENTATION_PROMPT.md
  PRE_ASTRA_CHECKLIST.md
```

Example initialization:

```bash
mkdir -p flyholdem
cd flyholdem
git init
# Copy the three Markdown files here.
git add flyholdem_build_spec.md ASTRA_IMPLEMENTATION_PROMPT.md PRE_ASTRA_CHECKLIST.md
git commit -m "Add FlyHoldem implementation brief"
```

## Downloads Astra should perform and verify

Astra should automate these rather than asking you to collect them:

- DOOMFLY source at a pinned Git commit, used as an MIT-licensed reference with attribution;
- official MaleCNS v1.0 annotation, neurotransmitter, and connectome-weight tables;
- selected neuron coordinates or skeletons needed for the visualization, fetched separately from the simulation graph;
- Python 3.11 virtual environment and locked dependencies;
- PokerKit and the chosen environment/test libraries;
- PyTorch and the conventional poker-teacher implementation;
- dashboard dependencies and browser-test tooling.

The 1.1 GB MaleCNS connection table is the essential large download. Avoid downloading the 12.7 GB synapse-point table, 6.8 GB partner table, or 2.7 GB per-synapse neurotransmitter table unless a specific validated feature requires them. The smaller aggregate annotation/neurotransmitter tables plus connection weights are sufficient for the first simulator. Download neuron skeletons selectively or as a separate optional visualization asset.

## Accounts and keys

- No OpenAI API key is required when Astra runs as a signed-in local Codex task.
- No commercial poker-solver account is required; the stronger teacher is trained locally from the same game environment.
- No casino, poker-site, payment, or real-money account should be connected.
- GitHub authentication is optional for local work and required only if you want Astra to push a repository or open a pull request.
- Public hosting credentials are not needed for the initial build. Add them only after the local dashboard and recorded replay work.

## Starting the task

Open the prepared local repository in Codex, select GPT-6 Astra with high or xhigh reasoning, paste the body of `ASTRA_IMPLEMENTATION_PROMPT.md`, and grant the scoped download/build permissions it requests.

The first expected outputs are the repository skeleton, dependency locks, fixture demo, and data provenance registry. The full MaleCNS download and training should happen only after the data-free tests and small conditioning gate work.
