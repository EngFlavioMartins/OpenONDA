forAll(UBf, patchi)
{
   if (isA<timeVaryingInletOutletFvPatchField<vector>>(UBf[patchi]))
   {
       refCast<timeVaryingInletOutletFvPatchField<vector>>(UBf[patchi]).updateRefValue
       (
           inletRefValue
       );
   }
}
