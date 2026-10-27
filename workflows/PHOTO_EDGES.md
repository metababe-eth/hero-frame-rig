# Uploaded photograph focus edges

New developments of uploaded stills with **Subject layer off** use the explicit
`core-round-gaussian-photo-v2` rendering policy. Generated stills, held-subject
branches, existing saved takes and clip rendering retain their recorded policy.

V2 uses the raw saved disparity and five saved blur-weight PNGs. Foreground weights
spill smoothly over a finite Gaussian support instead of the legacy twelve hard
depth dilations. Ordinary photographic highlights are decoded to linear light
without the synthetic HDR expansion that made white packaging bleed into a haze.
The existing five Gaussian blur kernels, stock, grain and finishing chain remain.

The recipe records the V2 marker, highlight policy, dimensions and weight policy.
All five mask files travel with it; replay does not silently recompute masks with
a later algorithm. The desktop replay guard recognizes V1 and V2, including zero
blur, and requires the exact bounded-highlight inputs for V2. V2 layered graphs
are rejected until that branch is separately evaluated.

## Evidence and limits

The Royal Praline study reuses the real saved source/depth. Gentle 1x takes show
less white-box haze and a softer foreground silhouette. The 3.9x near take still
has conspicuous blending where estimated depth varies over the packaging; it is
not the preferred demonstration. Thin wire halos and same-plane depth errors
remain. This is a gather-blur approximation, not physical aperture rendering or
reconstruction of hidden background.

Bounded highlights are an explicit tone-policy change, even at zero blur. The
same zero-blur finishing pass and the V2 near pass agree exactly over 13,317
zero-focus pixels in the checked photo. This does not imply the whole image or
every label stays pixel-identical to the original after focus and grading.

Run `python -m unittest discover -s tests -p "test_photo*.py" -v` for the flat-plane,
focus, spill locality, saved-mask, version and replay-boundary checks. Private
visual comparison assets live outside this code repository.
