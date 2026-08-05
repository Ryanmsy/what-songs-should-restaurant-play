# Learning Log

Concepts and patterns explained during sessions, for reference.

## 2026-08-02 — Git (version control) vs. CI/CD (automation pipeline)

- **`git push`** moves your local commits to the remote repo (GitHub). That's it — it doesn't run anything by itself.
- **CI/CD** (Continuous Integration / Continuous Deployment) is a separate, *automated* system (e.g. GitHub Actions) that *reacts* to events like a push or a pull request — running tests, linting, builds, or deploys without manual intervention.
- This repo currently has no CI/CD configured (no `.github/workflows/` directory), so a `git push` right now only updates the remote branch — no checks run, nothing deploys automatically. Deploys (Render, Streamlit Cloud) are still a manual dashboard process.
- If/when CI/CD gets added: a YAML workflow file in `.github/workflows/` defines triggers (`on: push`, `on: pull_request`) and jobs (e.g. `pytest`, `ruff check`) that GitHub runs on its own servers.

## 2026-08-02 — Why a commit on one branch doesn't appear on another

- A branch is a pointer to a commit. That commit holds a full snapshot of every tracked file at that point in history.
- Branching (`git checkout -b new-branch`) copies the snapshot from whatever commit you branched *from* — not from any other branch.
- Two branches that diverged (e.g. `main` and `like-dislike-feedback`) have separate timelines. A commit made on one only extends that branch's timeline; it does not touch the other, even if they share an early common ancestor.
- A fix becomes visible on other branches only when it's explicitly brought over: `git merge`, `git rebase`, or `git cherry-pick`.
- Easy to confuse with: *uncommitted* working-directory changes DO follow you across `git checkout` to another branch (if there's no conflict) — but once something is committed, it's locked to its branch's history until merged. This is why a `.gitignore` fix committed only on `like-dislike-feedback` didn't show up on a new branch cut from `main` — `main` never had that commit.

## 2026-08-05 — Notebooks → `src/` refactor: separation of concerns, DRY, and entry-point types

- **DRY (Don't Repeat Yourself) / duplication smell:** `deployment_draft/artifact/build_artifact.py` explicitly says in its own docstring that it "reproduces the pipeline from `notebooks/05b_recommender.ipynb`... as a plain script." Same recommendation logic exists twice by hand — a bug fix in one place doesn't propagate to the other. This is the concrete signal that logic belongs in one importable module, not copy-pasted.
- **`sys.path` hacking vs. real packages:** `deployment_draft/api/api_v2.py` does `sys.path.insert(0, ...scripts...)` before importing `feedback_store`. This works but is fragile (depends on relative file layout, breaks if the file moves). The proper fix is a `src/` layout installed with `pip install -e .` (an "editable install"), so imports work from an actual installed package rather than a relative path guess.
- **`src/` layout (Python packaging convention):** shared logic (cleaning, PCA feature transforms, the recommender scoring function) belongs in an importable package under `src/`. Notebooks and scripts then `import` from it instead of each holding their own copy.
- **Script vs. Service vs. Library — three different execution models:**
  - *Script* (`build_artifact.py`): runs top-to-bottom once, then exits.
  - *Service* (`api_v2.py`, run by uvicorn/Render): long-running, event-driven, never "finishes."
  - *Library* (a `src/` package): has no execution model at all — it only gets imported by other things.
  - Lesson: a single `main.py` that "runs everything" doesn't fit here, because this project already has three independent runtime processes (artifact build, API service, Streamlit client) each invoked separately in production (Render calls `api_v2.py`, Streamlit Cloud calls `app.py` directly — neither calls a `main.py`). The right move is extracting shared logic into `src/`, not collapsing the entry points into one.
  - Notebooks keep their proper role (EDA, inline plots, fast iteration) and become thin clients that import from `src/`, rather than being replaced outright.
