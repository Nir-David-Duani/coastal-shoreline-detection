# Algorithm

Yam Yabasha uses one intentionally simple, inspectable approach: a semantic
water/land boundary is detected in two dimensions, while a user hint says only
where that boundary is relevant.

## Inputs per camera location

Each location needs:

- `reference.jpg` — the fixed coordinate system.
- `shoreline_hint.json` — coarse points clicked once on the reference.
- Optional `reference_roi_mask.png` — stable scenery used only for registration
  feature detection.

The shoreline hint is not a label and does not need to trace the true shoreline.
Its only purpose is to create a generous search corridor.

## Processing stages

### 1. Registration

SIFT features are detected in the reference and moving frame. Descriptor matches
pass Lowe's ratio test, and RANSAC estimates a moving-to-reference homography.
Weak transforms are rejected using:

- number of inliers,
- inlier ratio,
- mean inlier reprojection error.

The transform is applied to the entire frame. After alignment, the saved hint
has the same coordinates for every frame from that camera.

### 2. Semantic prompt maps

CLIPSeg predicts one full-resolution probability map for each text prompt.
Water prompts and land prompts are combined separately:

`Pwater = combine(water prompt maps)`

`Pland = combine(land prompt maps)`

Using separate positive concepts is preferable to asking the model directly for
"shoreline": the actual output we need is the interface between two semantic
regions.

### 3. Water/land contrast

The signed contrast is:

`C = Pwater - Pland`

Positive values support water; negative values support land. Gaussian smoothing
and a configurable threshold produce a full-image water region. Morphological
closing and opening remove small gaps and isolated fragments.

### 4. Boundary evidence

Two independent cues are multiplied:

1. The spatial gradient magnitude of `C` — a rapid semantic transition.
2. Nearby support from both classes — high water and land probabilities in a
   local neighborhood.

This suppresses edges that have visual contrast but do not separate plausible
water from plausible land.

### 5. Full-image 2D contours

All connected contours are extracted from the full water region. There is no
assumption that the shoreline is horizontal, single-valued, straight, or
monotonic. Curved and locally vertical segments are valid.

### 6. ROI clipping

Only after contour extraction, contour segments outside the user corridor are
removed. This order is essential: masking the water map first would manufacture
an artificial contour along the corridor boundary.

### 7. Candidate selection

Short contour runs are rejected. Each remaining candidate receives:

`score = mean boundary evidence + length_weight × normalized length`

The highest-scoring connected run is selected and median-smoothed. The output
is an ordered `(x, y)` polyline in reference-image coordinates.

## What is deliberately not used

- No `side` parameter.
- No global shoreline direction.
- No row/column transects.
- No greedy path.
- No dynamic programming path.

Those approaches force geometric assumptions that are unreliable when a coast
curves or changes orientation.

## Failure safety

By default, inference stops if registration quality is insufficient. Applying a
reference-frame hint to an unaligned frame can return a plausible-looking but
spatially wrong line, so silent fallback is unsafe.

An empty candidate list is also a valid diagnostic result. It means no connected
semantic water/land boundary met the current ROI and filtering constraints.
