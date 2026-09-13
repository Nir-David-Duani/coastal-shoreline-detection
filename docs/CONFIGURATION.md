# Configuration reference

Configuration is defined by immutable dataclasses in
`src/yam_yabasha/config.py`. Use `dataclasses.replace` to change selected
values without duplicating all defaults.

## Registration

- `feature_type`: `sift` is the default; `orb` and `akaze` are also supported.
- `transform_type`: `homography` for projective camera motion, or `affine`.
- `ratio_test`: strictness of descriptor ambiguity filtering. Lower is stricter.
- `ransac_threshold_px`: maximum reprojection residual considered an inlier.
- `min_inliers`: minimum geometric support for accepting alignment.
- `min_inlier_ratio`: rejects transforms supported by a small match minority.
- `max_reprojection_error_px`: final mean inlier error safety limit.

## Segmentation

- `model_name`: Hugging Face CLIPSeg checkpoint.
- `water_prompts`, `land_prompts`: semantic concept ensembles.
- `water_combine`, `land_combine`: `max` includes any strong prompt; `mean`
  requires broader agreement.
- `device`: `None` automatically chooses CUDA when available.

## ROI contour

- `corridor_radius_px`: width of the search area around the clicked hint.
  The current default is `120` px; increase it when shoreline position varies
  substantially.
- `contrast_threshold`: water is defined where `Pwater - Pland` equals or exceeds this
  value. The validated default is `-0.05`; increase it to require stronger
  water preference.
- `blur_sigma`: smooths the semantic contrast before thresholding.
- `morphology_kernel`: spatial size for closing/opening.
- `close_iterations`: connects small breaks in the water region.
- `open_iterations`: removes small isolated water fragments.
- `semantic_radius_px`: neighborhood in which both water and land support are
  sought.
- `min_run_points`: minimum contour sample count.
- `min_run_length_px`: minimum geometric candidate length.
- `length_weight`: candidate-score contribution from coverage.
- `smoothing_window`: median window on the final ordered polyline.
- `border_margin_px`: excludes image-edge contours.

## Recommended tuning order

1. Inspect registration before tuning shoreline extraction.
2. Tune prompts using the separate water and land maps.
3. Adjust `contrast_threshold`.
4. Make the hint corridor broad enough to cover realistic movement.
5. Adjust morphology only when regions have visible holes or fragments.
6. Tune minimum run length and `length_weight` if multiple candidates remain.

Change one group at a time. Use the runtime six-stage `pipeline.png` for quick
inspection and the notebook or README eight-stage figures for deeper analysis,
instead of judging only the final red line.
