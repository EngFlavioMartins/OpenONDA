volVectorField gradP = -(fvc::ddt(U) + fvc::div(phi, U) - fvc::laplacian(nu, U));

forAll(pBf, patchi)
{
   if (isA<fluxControlledPressureFvPatchScalarField>(pBf[patchi]))
   {
       vectorField normals = mesh.Sf().boundaryField()[patchi]/mesh.magSf().boundaryField()[patchi];

       refCast<fluxControlledPressureFvPatchScalarField>(pBf[patchi]).updateCoeffs
       (
           gradP.boundaryFieldRef()[patchi] & normals
       );
   }
}
