# Tips:

Note that there are two meshes that can be used in this OpenFOAM case, namely:

```./constant/polyMesh_from_blockMesh.backup```

and

```./constant/polyMesh_from_cfMesh.backup```

To prepare the case for the simulation, just rename one of the files above to ```./constant/polyMesh.orig``` using, for example:

``` cp -r ./constant/polyMesh_from_cfMesh.backup /constant/polyMesh.orig ```

To produce the mesh using cfMesh such as done for this test case:

1. Create the sphere using Blender. Export it as a stl file.
2. Open the stl file and replace its first and last lines where it says "Exported from Blender ..." by "Domain"
3. Run cfMesh (notice that the cfMesh configuration files are already stored at ``` ./system/meshDict```) using ```cartesianMesh```
4. Run ```renumberMesh -overwrite```

Done!

To produce the mesh using blockMesh, simply run the command ``` blockMesh``` in your terminal from this directory.