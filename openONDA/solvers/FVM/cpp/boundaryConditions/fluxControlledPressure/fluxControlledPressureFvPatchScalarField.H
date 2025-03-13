/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2016 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::fluxControlledPressureFvPatchScalarField

Group
    grpInletBoundaryConditions grpWallBoundaryConditions

Description
    This boundary condition sets the normal pressure gradient equals to the flux
    based on all velocity terms of the Navier-Stokes equation:

        grad(p) & n = -(ddt(U) + div(phi, U) - laplacian(nu, U)) & n

    Example of the boundary condition specification:
    \verbatim
    <patchName>
    {
        type            fluxControlledPressure;
    }
    \endverbatim

See also
    Foam::fixedGradientFvPatchField

SourceFiles
    fluxControlledPressureFvPatchScalarField.C

\*---------------------------------------------------------------------------*/

#ifndef fluxControlledPressureFvPatchScalarFields_H
#define fluxControlledPressureFvPatchScalarFields_H

#include "fvPatchFields.H"
#include "fixedGradientFvPatchFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
             Class fluxControlledPressureFvPatchScalarField Declaration
\*---------------------------------------------------------------------------*/

class fluxControlledPressureFvPatchScalarField
:
    public fixedGradientFvPatchScalarField
{
    // Private data

        //- Current time index (used for updating)
        label curTimeIndex_;


public:

    //- Runtime type information
    TypeName("fluxControlledPressure");


    // Constructors

        //- Construct from patch and internal field
        fluxControlledPressureFvPatchScalarField
        (
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&
        );

        //- Construct from patch, internal field and dictionary
        fluxControlledPressureFvPatchScalarField
        (
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&,
            const dictionary&
        );

        //- Construct by mapping given fluxControlledPressureFvPatchScalarField onto
        //  a new patch
        fluxControlledPressureFvPatchScalarField
        (
            const fluxControlledPressureFvPatchScalarField&,
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&,
            const fvPatchFieldMapper&
        );

        /////////////////////////////////////////// THESE WERE NOT HERE
        fluxControlledPressureFvPatchScalarField
        (
            const fluxControlledPressureFvPatchScalarField&
        ) = delete;

        //- Copy constructor setting internal field reference
        fluxControlledPressureFvPatchScalarField
        (
            const fluxControlledPressureFvPatchScalarField&,
            const DimensionedField<scalar, volMesh>&
        );
        //- Construct and return a clone setting internal field reference
        virtual tmp<fvPatchScalarField> clone
        (
            const DimensionedField<scalar, volMesh>& iF
        ) const
        {
            return tmp<fvPatchScalarField>
            (
                new fluxControlledPressureFvPatchScalarField(*this, iF)
            );
        }


    // Member functions

        //- Update the patch pressure gradient field from the given snGradp
        virtual void updateCoeffs(const scalarField& snGradp);

        //- Update the patch pressure gradient field
        virtual void updateCoeffs();

        //- Write
        virtual void write(Ostream&) const;
};


}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


#endif

// ************************************************************************* //
