forAll(pBf, patchi)
{
   if (isA<timeVaryingOutletInletFvPatchField<scalar>>(pBf[patchi]))
   {
       refCast<timeVaryingOutletInletFvPatchField<scalar>>(pBf[patchi]).updateRefValue
       (
           outletRefValue
       );
   }
}

forAll(pBf, patchi)
{
   if (isA<timeVaryingInletOutletFvPatchField<scalar>>(pBf[patchi]))
   {
       vectorField normals = mesh.Sf().boundaryField()[patchi]/mesh.magSf().boundaryField()[patchi];

       refCast<timeVaryingInletOutletFvPatchField<scalar>>(pBf[patchi]).updateRefGradient
       (
           inletRefGradient & normals
       );
   }
}
