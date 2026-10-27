# Reopen a saved image recipe

In **Use my image → Open a saved image recipe**, choose a Hero Frame Rig image
recipe ZIP and select **Open image recipe**. Opening restores the working source,
cached depth/disparity and subject mask, every saved take, selected take, focus,
aperture, look, depth-of-field mode, subject switch, Guide setting and source
metadata. Saved click markers are restored when recorded; old recipes do not gain
invented click locations. The byte-for-byte original is retained when bundled.

Opening submits no backend job and executes no graph. Comparing saved takes and
saving another recipe also need no backend. A subsequent focus/develop action uses
the current application renderer with the cached maps; it creates another take
and requires the selected backend. Old hero images and their recorded graphs are
preserved. Their appearance does not change when the renderer changes.

`image_recipe.load_image_recipe` is the untrusted image-recipe boundary. The pinned
NOLA study loader remains separate and still requires its registered path and hash.
Clip recipes continue to use their existing importer.

## Validation and compatibility

- Image schemas 1 and 2 are accepted. Schema 2 requires the complete SHA-256
  manifest. Schema 1 without a manifest opens with an explicit legacy message;
  when selection was not recorded it selects the last saved take. Checksums prove
  package consistency, not authorship or trusted content.
- Saved graphs are bounded API-format DAGs with valid connections. LoadImage paths
  must name bundled PNGs. The loader does not import node code, execute graphs,
  contact ComfyUI, or fetch resources. This validation is not permission to execute
  an arbitrary saved graph elsewhere.
- ZIP path traversal, Windows path aliases/device names, symlinks/special files,
  case/Unicode duplicate paths, encrypted members, unsupported compression and
  conflicting file/directory paths are rejected before extraction. Directory entry
  count is checked before constructing the ZIP member list. ZIP64/spanned archives
  and directory records are not needed by app exports and are unsupported.
- Limits: 256 MiB compressed/expanded, 64 MiB per member, 4 MiB per JSON document,
  512 members, 64 takes, 2,048 nodes per graph, 16 million pixels per working asset,
  64 million pixels for the preserved original, and 300 million decoded working
  pixels across referenced images. PNGs are decoded and must match the working
  frame. EXR headers are bounded before full decoding; depth must be finite,
  nonnegative floating-point data. Original hashes, bytes, dimensions, encoding
  and EXIF orientation must match provenance.
- JSON duplicate keys, non-finite numbers, invalid selection/settings, missing
  assets and malformed images/depth/graphs fail with an actionable error. Files
  are placed in a new private temporary session directory. Failed imports remove
  only the new directory they created; existing destinations are never cleared.
- Re-export retains frame/take provenance, original media, saved settings/graphs,
  optional generated format/resolution and click points. It records the imported
  archive's SHA-256 and deduplicates shared input maps. Sparse take numbers advance
  from the highest existing take, so continuing an edit does not overwrite history.

## Offline checks

Run `python -m unittest discover -s tests -p test_image_recipe_open.py -v` using the
project environment. Tests cover a real save/open/save cycle, UI output restoration,
backend isolation, sparse IDs and continued development with a mocked job result,
legacy archives, path aliases/symlinks, ZIP limits, checksum failures, missing and
invalid settings/graphs/images/depth, and preservation of existing directories.

When the private evidence folder exists beside `brief32`, the suite additionally
opens and re-saves `own-media-20261023/photo_demo/ROYAL_PRALINE_FIVE_TAKES_recipe.zip`.
It checks all five saved settings/jobs, selection 5, and exact original/working/
depth/mask/hero bytes. The private image fixture is not part of the code repository.
These checks do not claim new inference, desktop graph replay, or visual approval.
