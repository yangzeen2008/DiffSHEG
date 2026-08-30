# Face model assets

The primary thesis-rendering asset is
`microsoft_rocketbox_female_adult_01_arkit/`. It is a Blender-compatible FBX
with a full body rig, materials, and 52 ARKit facial blendshapes.

`threejs_facecap_arkit/` is kept only as a lightweight GLB reference for web
preview. Its GLB requires `EXT_meshopt_compression` and `KHR_texture_basisu`,
so it is not the default Blender input.

Do not place generated renders in this directory. Put rendered comparisons
under `bvh_output/` with the corresponding experiment output.
