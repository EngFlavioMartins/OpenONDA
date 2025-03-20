forAll(UBf, patchi)
{
   if (isA<timeVaryingInletOutletFvPatchField<vector>>(UBf[patchi]))
   {
       vectorField normals = mesh.Sf().boundaryField()[patchi]/mesh.magSf().boundaryField()[patchi];

       refCast<timeVaryingInletOutletFvPatchField<vector>>(UBf[patchi]).updateRefGradient
       (
           outletRefGradient
       );
   }
}
