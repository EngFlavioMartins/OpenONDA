forAll(pBf, patchi)
{
   if (isA<timeVaryingOutletInletFvPatchField<scalar>>(pBf[patchi]))
   {
       vectorField normals = mesh.Sf().boundaryField()[patchi]/mesh.magSf().boundaryField()[patchi];

       refCast<timeVaryingOutletInletFvPatchField<scalar>>(pBf[patchi]).updateRefGradient
       (
           inletRefGradient & normals
       );
   }
}

forAll(pBf, patchi)
{
   if (isA<timeVaryingInletOutletFvPatchField<scalar>>(pBf[patchi]))
   {
       refCast<timeVaryingInletOutletFvPatchField<scalar>>(pBf[patchi]).updateRefValue
       (
           outletRefValue
       );
   }
}
