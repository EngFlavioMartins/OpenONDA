- Create the geometries using blender. In this case, a sphere

- Export the geomery as stl files. Lets name it "sphere.stl"
    **Note:** don't forget to tick the format of your file to be "ASCII" in the "general" box and "selection only".

- Open the stl file in a text editor and rename the first and last lines, where it reads "solid xxxx" and "endsolid xxxx" to "solid sphere" and "endsolid sphere", respectively.

- Open cfMesh at the root directory. 
    **Note:** make sure there is a "0" folder (and not only the 0.orig), otherwise the solver output an error.

    - Run the command: ```surfaceGenerateBoundingBox ./meshing/sphere.stl ./meshing/sphere_in_domain.stl 1.5 1.5 1.5 1.5 1.5 1.5```. This will create an aerodynamic domain around the sphere geometry.

    - Run the command: ```cartesianMesh``` to generate a hex-dominant 3D mesh

- From the terminal execute the following:

    - Run the command: ```checkMesh``` to verify evertyhing.

    - Run the command ```renumberMesh -overwrite```. 
    
        **Note:** It reorders the mesh cells, faces, and points to reduce the bandwidth of the system matrix used in solving equations. The goal is to minimize indirect memory access and improve solver efficiency.

    - Copy the final mesh file to the default OpenONDAs directory: ```cp -r ./constant/polyMesh/ ./constant/polyMesh.orig```

