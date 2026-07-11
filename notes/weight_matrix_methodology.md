# How We Built the Weight Matrix (W) — Step by Step

This is the plain-language walkthrough of how `W` (the matrix that turns a restaurant's
vibe into a target song vibe) gets built in `notebooks/05b_recommender.ipynb`. Read this
before the notebook if the code feels like a lot at once.

## The problem in one sentence

Restaurants and songs live in two different, unrelated feature spaces (13 Yelp PCs vs.
6 Spotify PCs) — there's no natural way to compare "romantic ambience" to "danceability"
directly, so we need a bridge matrix `W` such that `S ≈ W · Y` (Yelp PC vector in,
Spotify PC vector out).

## Step by step

1. **Run PCA on both datasets separately** (notebook 04). This gives every restaurant a
   13-number Yelp-PC fingerprint and every song a 6-number Spotify-PC fingerprint.
   Read `pca_yelp.components_` / `pca_spotify.components_` to figure out, in plain
   English, what each PC actually represents (e.g. yelp `pc3` ≈ "fine dining", spotify
   `pc4` ≈ "fast tempo"). **Never assume a label — verify against the loadings.** An
   earlier pass mislabeled yelp `pc4` as "late-night" when the real dominant loading was
   noise level; the mistake was only caught by re-reading `components_` directly.

2. **Pick the 8 Yelp PCs that have a clean, human-readable dominant loading** (pc1–pc8).
   These become our 8 "archetypes" — sit-down dinner, brunch, fine dining, loud/chaotic,
   hipster, dive bar, late-night, good-for-groups.

3. **For each archetype, get a real target song vibe instead of guessing one.**
   This is the key idea: rather than hand-typing "fine dining should be tempo=85,
   energy=0.25," we pick ~10 well-known artists who obviously fit that archetype
   (Miles Davis / John Coltrane for "fine dining", Daft Punk / Disclosure for
   "good for groups"), pull every one of their real songs' actual Spotify-PC scores
   out of `song_pca.csv`, and average them. That average becomes the archetype's
   target vector. If an archetype doesn't have enough anchor-song matches
   (`< 5` songs), it falls back to a hand-authored audio-feature guess (danceability,
   tempo, etc.) projected through the real scaler + PCA pipeline instead of a guess in
   raw PC space — so it's at least on the right scale, if not anchor-grounded.

   > **Example** (toy numbers — real songs have 6 numbers each, not 2; this is
   > simplified so the arithmetic fits on one line). Say "fine dining" has 3 anchor
   > songs, and each song's music vibe is just two numbers, `[energy-ish, speechy-ish]`:
   > ```
   > Miles Davis  – "So What"      → [-0.8,  0.6]
   > Coltrane     – "Naima"        → [-1.0,  0.4]
   > Bill Evans   – "Waltz for D." → [-0.6,  0.5]
   > -------------------------------------------
   > average                       → [-0.8,  0.5]   <- this is "fine dining"'s target
   > ```
   > "Loud/chaotic" would go through the same process with its own anchor artists
   > (AC/DC, Foo Fighters, ...) and land somewhere very different, e.g. `[1.1, -0.2]`
   > — high energy, low speechiness. Every archetype ends up as one such vector.

4. **Build one training pair per archetype**: input = a one-hot vector (1.0 on that
   archetype's Yelp PC, 0 everywhere else), output = that archetype's target Spotify-PC
   vector from step 3.

   > **Example** (continuing the toy above, with just 2 archetypes instead of 8):
   > ```
   > input (which archetype?)   output (target music vibe)
   > fine dining:    [1, 0]  →  [-0.8,  0.5]
   > loud/chaotic:   [0, 1]  →  [ 1.1, -0.2]
   > ```
   > That's it — "training data" here just means two rows pairing a label with its
   > measured target. No restaurants or listening history involved yet.

5. **Solve for `W` with the normal equation.** With 8 training pairs like this,
   `W = argmin ||S - W·Y||²` has a closed-form solution — the same
   `W = S·Yᵀ·(Y·Yᵀ)⁻¹` from Gilbert Strang's course, done here via
   `sklearn.LinearRegression(fit_intercept=False)`. Because the inputs are one-hot,
   this collapses to something simple: **column `i` of `W` just becomes archetype
   `i`'s target vector.** The regression machinery isn't doing anything mysterious here
   — it's a convenient, correct way to assemble the columns, not a model discovering
   hidden structure from a large dataset.

   > **Example**: with the two training pairs above, `W` is just the two target
   > vectors placed side by side as columns:
   > ```
   >         fine-dining col   loud/chaotic col
   > W  =  [    -0.8              1.1        ]
   >       [     0.5             -0.2        ]
   > ```
   > Check it: `W @ [1, 0] = [-0.8, 0.5]` (exactly the fine-dining target) and
   > `W @ [0, 1] = [1.1, -0.2]` (exactly the loud/chaotic target). That's the whole
   > "fit" — with one-hot inputs there's no error to minimize, so the regression
   > just hands back the targets as columns. The real `W` is 6×8 (6 music traits,
   > 8 archetypes) instead of 2×2, built the same way.

6. **Sanity-check `W` against extreme real restaurants** ("boss-level eval"): take the
   real restaurant that scores highest on each Yelp PC, project it through `W`, and check
   if the top-3 nearest real songs make sense for that vibe. This is a spot-check, not a
   formal validation — see Limitations.

   > **Example**: say the most "fine dining" restaurant in the dataset isn't *purely*
   > fine dining — it also scores a little on loud/chaotic, e.g. its raw two-archetype
   > vibe score is `y = [3.6, 0.2]` (heavily fine dining, barely loud/chaotic).
   > 1. **Normalize** `y` to length 1: divide by
   >    `sqrt(3.6² + 0.2²) ≈ 3.61`, giving `y_normalized ≈ [0.998, 0.055]`.
   > 2. **Project through `W`**: `W @ y_normalized ≈ [-0.74, 0.49]`
   >    (using the toy `W` from step 5) — very close to the pure fine-dining target
   >    `[-0.8, 0.5]`, just nudged slightly by that small loud/chaotic component.
   > 3. **Find the closest real songs** to `[-0.74, 0.49]` in `song_pca.csv` by
   >    distance — those become the recommendations. If they're all moody jazz-like
   >    tracks, that's the sign `W` is behaving sensibly for this restaurant.

## Why this way, and not something else

- **Why not fit `W` on real listening data (restaurant → songs actually played there)?**
  That data doesn't exist yet. There's no dataset pairing real restaurants with the
  songs they play. Feedback data (thumbs up/down) is the eventual replacement for this
  whole step — see `notes/Recommendations_Engine_thought.md`.
- **Why not a fully hand-typed heuristic matrix** (just eyeball every cell of `W`)?
  Early drafts tried this and it's arbitrary and hard to defend — "why is `NoiseLevel`
  worth `-0.5` and not `-0.3`?" Anchoring to real songs by real artists replaces that
  arbitrary judgment with a measurement, even if the *choice of artists* is still a
  judgment call.
- **Why not a shared latent "vibe space" (two-tower / canonical correlation) instead of
  a direct Yelp-PC → Spotify-PC map?** More principled, but needs either labeled
  vibe data or a lot more engineering than an MVP justifies. Direct archetype anchoring
  gets a working, explainable baseline shippable now; the fancier approach is a
  reasonable next step once feedback data exists.
- **Why the normal equation instead of gradient descent?** With only 8 training rows and
  a closed-form solution available, there's no reason to iterate — the normal equation
  is exact and instant.

## Limitations (be upfront about these)

- **This is a hand-authored hypothesis, not a model fit to real preferences.** Nothing
  in this pipeline has ever been validated against an actual human saying "yes, that
  song works in that restaurant." The boss-level eval only checks whether results look
  *plausible*, not whether they're *correct*.
- **Only 8 of 13 Yelp PCs are covered.** pc9–pc13 don't have a clean, one-sentence
  dominant loading, so restaurants are effectively recommended using only part of their
  profile.
- **Half the anchor-artist lists were never reviewed by a human.** `pc2`, `pc3`, `pc5`,
  `pc8` came from a curated list; `pc1`, `pc4`, `pc6`, `pc7` are placeholders that were
  never checked (see `CLAUDE.md` → Deferred). Some placeholder artists (Michael Bublé,
  Vance Joy, Guns N' Roses, The Rolling Stones, CCR) have zero matches in the song
  catalog because it's filtered to 2000–2020 releases — those archetypes silently fall
  back to the weaker hand-authored guess from step 3 instead of a real anchor.
- **8 training points is a very small sample** to fit even a small (6×13) matrix
  against. There's no held-out test set — "does this look right" is a human judgment
  call on a handful of boss-level restaurants, not a metric.
- **The hub-penalty threshold (5% of restaurants) and any future feedback-driven
  update rate are unvalidated defaults**, not tuned against real usage.
- **One-hot training inputs mean `W`'s columns are independent of each other** — the
  model never learns anything about how archetypes *combine* for a restaurant that's,
  say, half fine-dining and half loud/chaotic. Real restaurants aren't one-hot, so this
  is an extrapolation the model was never actually trained to make.
