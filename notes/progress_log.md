# Progress Log

Running journal of what changed and why, at a decision level — not a commit log (git
already has that) and not a tutorial (notebooks already have that). Framework per
entry: **Steps** (what happened, ~2 sentences) → **Why** → **Considerations** →
**Remember** (optional).

---

## 1. Initial MVP (main branch)

**Steps:** Built the recommender on the original data: `tracks_features.csv` (1.2M
Spotify tracks, audio features only) and `yelp_clean.csv` (52,268 restaurants,
ambience/mood attributes only). PCA on both, hand-authored weight matrix `W`
(Yelp-PC → Spotify-PC) grounded in real anchor-artist songs, deployed as a
FastAPI + Streamlit MVP.

**Why:** No real listening-history data exists to fit `W` on, so anchor artists stood
in as a supervision signal — a documented hypothesis, not a trained model.

**Considerations:** Two known gaps from day one: no song popularity (can't tell
obscure from well-known), and no restaurant cuisine/category signal at all.

**Remember:** Full math/reasoning lives in `notes/weight_matrix_methodology.md`.

---

## 2. Spotify live API investigation — dead end

**Steps:** Tried pulling real popularity data directly from the Spotify Web API to
fix the "obscure song" gap. Found Spotify locked `popularity`, `audio-features`,
and batch `/tracks` behind Extended Quota Mode approval (a Nov-2024 policy change)
— confirmed with real 403s / stripped fields against our own app, not assumption.

**Why:** A live pull would've been strictly worse than what we already had — no
audio features *and* no popularity, for any newly-created developer app.

**Considerations:** This path is closed unless Spotify grants Extended Quota
approval, which isn't realistic for a personal project on this timeline.

---

## 3. Switched to a new Spotify dataset

**Steps:** Adopted `maharshipandya/spotify-tracks-dataset` (114K tracks, mirrored on
Hugging Face) — has real `popularity` and a `track_genre` label the original never
had. Cleaned/deduped/ran PCA on it (notebook `01b`): 114K → 81,198 rows after
dropping nulls/`tempo==0`/duplicates, 6 PCA components at 85% variance (same
component count as the original).

**Why:** Smaller catalog, but genuine popularity + genre data beats a bigger catalog
with neither.

**Considerations:** No release-date/year column at all in this dataset — lost the
"filter to 2000–2020" quality control the original pipeline had.

**Remember:** Created branch `spotify-dataset-migration` off `building_api` to keep
this separate from the API/OAuth work already in progress there.

---

## 4. Found the real cause of the "Texas BBQ" problem

**Steps:** Restaurants named e.g. "Texas BBQ" got no country-music weighting.
Root cause: the Yelp side has **zero cuisine/category signal**, not a song-side gap
— confirmed by checking `yelp_ml_ready.csv`'s columns directly (ambience/mood only).

**Why:** `W` can't map a restaurant to country music if nothing in the restaurant's
feature vector says "this is a BBQ/Southern place" in the first place.

**Considerations:** Fixing this needed the original Yelp Academic Dataset's
`business.json` (has a `categories` field) — a second data-source hunt, not a
song-side fix.

**Remember:** All 52,268 restaurants matched the raw `business.json` by
`business_id` — no coverage loss on the join.

---

## 5. Curated cuisine categories, rebuilt Yelp PCA

**Steps:** Mapped ~700 raw category tags down to 17 curated cuisine buckets
(user-reviewed: Pizza folded into American, American Traditional/New merged,
Steakhouse+Barbeque merged). Re-ran Yelp PCA with cuisine included (notebook `02b`)
— went from 13 to 25 components at 85% variance.

**Why:** Cuisine tags are mostly independent of ambience/mood features and of each
other, so they don't compress the way correlated features do — needing almost double
the components is expected, not a sign PCA "got worse."

**Considerations:** Some cuisines share/entangle across components (Indian split
across two PCs; SteakhouseBBQ never got its own clean axis) — real, not a bug.

---

## 6. First cuisine-archetype attempt failed — and why

**Steps:** Tried adding 5 cuisine-driven PCA archetypes, grounded by averaging each
genre's songs into a centroid and doing nearest-neighbor search in audio-feature
space (same method that worked for artist-anchoring). Tested directly: even a
*single pure genre's* centroid (`country` alone) returned mostly unrelated genres as
nearest neighbors.

**Why it failed:** Audio features (danceability/energy/tempo/etc.) describe how a
song *sounds*, not what genre it is — two songs can match on those numbers while
being musically unrelated. Genre isn't recoverable from this feature space.

**Considerations:** This is a real limitation of the dataset/features, not something
more tuning would fix — a different mechanism was needed entirely.

---

## 7. Fixed: genre as a hard filter, not a PCA target

**Steps:** Redesigned so genre restricts the *candidate pool* directly from each
restaurant's raw `Cuisine.*` flags, before ranking by vibe. `W` went back to just the
8 original vibe archetypes, which now rank *within* the genre-filtered pool instead
of trying to reproduce genre from audio features.

**Why:** Guarantees genre-consistency by construction instead of hoping the math
finds it — validated directly (BBQ/Southern → blues/gospel/country family, Mexican →
salsa/spanish, Japanese → j-pop/j-rock, Indian → indian).

**Considerations:** Only 9 of 17 cuisine buckets have a defensible genre mapping in
this catalog's 114 genres (Italian, Greek, Thai, Vietnamese, Cajun/Creole, Asian
Fusion, generic American don't) — those fall back to pure vibe-matching, a real gap.

---

## 8. Added the popularity penalty, built the v2 artifact

**Steps:** Added an obscure-song penalty (same `sqrt`-scaled shape as the old
hub-penalty, inverted to penalize *low* popularity instead of *high* frequency) —
8,520/81,198 songs affected (popularity < 10). Bundled everything into
`recommender_artifact_v2.pkl` (W, cuisine filters, popularity penalty, song +
restaurant metadata).

**Why:** This was the other half of the reason we switched datasets in the first
place — real popularity data instead of the old anchor-artist-list proxy.

**Considerations:** User wants both a threshold penalty (shipped) *and* a
continuous log-weighted alternative documented side-by-side for the math comparison
— not yet built.

**Remember:** Saved as a *new* file alongside the old `recommender_artifact.pkl`, not
a replacement — API will switch between them via an env var, not a hard cutover.

---

## 9. Verified API baseline before touching it

**Steps:** Confirmed `api_v2.py` still works end-to-end against the *old* artifact
(`/health`, `/restaurants`, `/recommend` all responded correctly) before making any
changes for v2 support.

**Why:** Cheap checkpoint to isolate any future bugs to the v2 wiring, not leftover
breakage from earlier work.

---

## 10. Built the popularity math-comparison notebook

**Steps:** Added `05e_popularity_penalty_comparison.ipynb` — threshold (shipped) vs.
continuous log-weighted penalty, side by side: formulas, a value table, a plot, and a
real restaurant's top-5 under each. Caught and fixed a real bug along the way
(`log1p(0) == 0` caused divide-by-zero in the continuous formula at
`popularity == 0`).

**Why:** Documents the tradeoff for the alternative that wasn't shipped, not just the
decision.

**Considerations:** Continuous touches ~100% of songs (always active); threshold
touches ~10%. Threshold shipped because its effect is easy to reason about.

---

## 11. Wired `api_v2.py` to support both artifacts

**Steps:** Added small helper functions that translate between the old and new
artifact's differently-named fields (`yelp_pc_cols`/`archetype_pcs`,
`scaler_song`/`song_scaler`, `penalty`/`popularity_penalty`) so one `/recommend`
codepath serves either. Cuisine-genre filtering and popularity fields are additive on
the response — default to empty/`null`, so old-artifact behavior is provably
unchanged (tested byte-identical against the old artifact before/after).

**Why:** "Keep both artifacts, switch via config" (user's call) meant the API code
itself needed to handle either shape, not just the artifact files coexisting.

**Considerations:** Found the v2 artifact was missing `built_at` and
`spotify_pc_cols` (needed by `/health` and the dominant-PC lookup) — fixed in `05d`'s
save step and regenerated it, rather than patching around the gap defensively only.

**Remember:** Switch via `ARTIFACT_PATH=data/processed/recommender_artifact_v2.pkl`
env var. Verified live via curl: old artifact unchanged, new artifact's cuisine
filter matches the notebook exactly (BBQ/Southern restaurant → gospel/blues/bluegrass
only), and the no-cuisine-tag fallback correctly searches the full catalog.

---

## 12. Streamlit UI updated and verified in a real browser

**Steps:** Added a "Cuisine match — filtered to: ..." caption and a per-song
`(genre, popularity)` detail to the results view — both additive, hidden entirely
when absent (old artifact). Verified with an actual headless-browser run (Playwright)
against both live servers, not just a code read: searched "Smokin'", got the BBQ/
Southern restaurant, clicked Recommend, screenshotted the rendered page.

**Why:** UI changes need to be seen rendering, not just reasoned about — confirmed
the em-dash renders correctly and the genre/popularity details appear exactly where
expected.

**Considerations:** No project-specific `run` skill existed yet for this repo: used
the generic server + Playwright pattern instead (fresh ports to avoid clashing with
earlier test servers still lingering from prior sessions).

**Remember:** Zero console errors in the browser; API log showed clean 200s for
`/health`, `/restaurants`, `/recommend`. Test servers stopped after verification.

---

## Not done yet (as of this entry)

- Decide whether `spotify-dataset-migration` is ready to merge back toward `main`/`building_api`, or needs more work first
- The 8 cuisine buckets with no genre mapping (Italian, Mediterranean, Greek, Thai, Vietnamese, Cajun/Creole, Asian Fusion, generic American) still fall back to pure vibe-matching — unresolved gap, not a bug
- Commit the branch's work (currently uncommitted)
