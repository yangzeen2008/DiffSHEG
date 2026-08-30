# Microsoft Rocketbox ARKit avatar

Primary 3D face/body asset for DiffSHEG expression validation.

## Contents

- `Female_Adult_01_facial.fbx`: rigged avatar with 52 ARKit blendshapes.
- `f001_*`: all seven textures referenced by the FBX.
- `arkit_channel_map.json`: deterministic mapping from the 51 DiffSHEG face
  coefficients to Rocketbox shape keys. `AK_52_TongueOut` is intentionally
  unused because the BEAT/DiffSHEG expression vector has 51 coefficients.
- `Female_Adult_01_preview.png`: upstream preview.
- `LICENSE.md`: upstream Microsoft Rocketbox MIT license.
- `HeadboxFaceMapper.asset`: upstream Unity Face Capture mapping reference.
- `rocketbox_arkit_ready.blend`: verified textured Blender scene with the
  canonical front camera and neutral lighting.
- `rocketbox_arkit_clay.blend`: texture-free face-evaluation scene. It keeps
  the same mesh and Shape Keys, shows the neutral clay head/neck, and hides
  clothing plus texture-card hair.
- `rocketbox_arkit_clean_clay.blend`: preferred face-evaluation scene. It keeps
  the neutral clay head and separates the two eyeball islands from the shared
  head material so only the eyeballs retain the original iris/sclera texture.
  All combined opacity cards, including the Rocketbox eyelash cards, are hidden:
  those cards use the hair alpha atlas and can drift away from the deformed
  eyelid under large ARKit eye motions. This avoids both hollow-looking eyes and
  white triangular card intersections around the eyelids.

## Verification

Verified on 2026-08-30:

- FBX header: Kaydara binary FBX.
- File SHA-256:
  `2E6BA71FB83A6899E3D4DD35DA07FB5B078EEF0C4D593F73D70CFC05DC4644CB`.
- Shape keys found in the FBX: `AK_01_BrowDownLeft` through
  `AK_52_TongueOut` (52/52).
- DiffSHEG semantic channels matched: 51/51, in the same order as
  `FACE_COEFFICIENTS` in `assets/analyze_full_validation_transitions.py`.
- All texture filenames referenced by the FBX are present beside the FBX.

## Blender use

Import `Female_Adult_01_facial.fbx` with Blender's FBX importer. Keep the FBX
and textures together so material paths resolve. DiffSHEG coefficients are
already in the expected normalized range; assign each value, clipped to
`[0, 1]`, to the target shape key listed in `arkit_channel_map.json`.

For a fair GT/prediction comparison, first render coefficients without extra
temporal smoothing. Any smoothing variant should be rendered and labelled as
a separate post-processing condition.

`assets/render_rocketbox_expression_video.py` applies a Rocketbox-specific,
render-only `rocketbox_safe` transfer by default: `eyeWideLeft/Right` are scaled
by `0.25` and capped at `0.20`. The verified mesh develops a detached dark upper
eyelid crease above that range even though the source coefficients are valid.
The source JSON is never changed, and the render report records every adjusted
frame/channel. Use `--eye-safety-profile raw` when a literal coefficient render
is required for diagnostics or a separately labelled raw comparison.

The reusable Blender CLI entry points are
`assets/prepare_rocketbox_blender_scene.py` and
`assets/prepare_rocketbox_clay_variant.py`. The clay builder verifies 51/51
DiffSHEG channels. Without `--keep-eye-cards` it removes all file-backed
texture use; with `--keep-eye-cards` it retains only the original eye/eyelash
alpha map. Add `--keep-eye-texture` to preserve the original material only on
the two disconnected eyeball islands. Both splits and their polygon counts are
recorded in the generated report. The preferred evaluation build uses
`--keep-eye-texture` without `--keep-eye-cards`; card retention remains an
explicit diagnostic option.

## Source, attribution, and license

- Model repository: https://github.com/microsoft/Microsoft-Rocketbox
- Facial toolkit/reference: https://github.com/openVRlab/Headbox
- Rocketbox paper: https://doi.org/10.3389/frvir.2020.561558
- HeadBox paper: https://www.microsoft.com/en-us/research/publication/headbox-a-facial-blendshape-animation-toolkit-for-the-microsoft-rocketbox-library/

The Rocketbox repository distributes the avatar library under the MIT License;
the exact upstream license text is retained as `LICENSE.md`. Preserve this
notice when redistributing the asset.
