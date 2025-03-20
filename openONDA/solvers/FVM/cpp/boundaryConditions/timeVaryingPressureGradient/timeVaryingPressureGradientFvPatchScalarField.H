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
    Foam::timeVaryingPressureGradientFvPatchScalarField

Group
    grpInletBoundaryConditions grpWallBoundaryConditions

Description
    This boundary condition receives from an external solver an array of values
    for imposing a Neumann boundary condition for the pressure field.

    Example of the boundary condition specification:
    \verbatim
    <patchName>
    {
        type            timeVaryingPressureGradient;
    }
    \endverbatim

See also
    Foam::fixedGradientFvPatchField

SourceFiles
    timeVaryingPressureGradientFvPatchScalarField.C

\*---------------------------------------------------------------------------*/

#ifndef timeVaryingPressureGradientFvPatchScalarFields_H
#define timeVaryingPressureGradientFvPatchScalarFields_H

#include "fvPatchFields.H"
#include "fixedGradientFvPatchFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
             Class timeVaryingPressureGradientFvPatchScalarField Declaration
\*---------------------------------------------------------------------------*/

class timeVaryingPressureGradientFvPatchScalarField
:
    public fixedGradientFvPatchScalarField
{
    // Private data

        //- Current time index (used for updating)
        label curTimeIndex_;


public:

    //- Runtime type information
    TypeName("timeVaryingPressureGradient");


    // Constructors

        //- Construct from patch and internal field
        timeVaryingPressureGradientFvPatchScalarField
        (
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&
        );

        //- Construct from patch, internal field and dictionary
        timeVaryingPressureGradientFvPatchScalarField
        (
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&,
            const dictionary&
        );

        //- Construct by mapping given timeVaryingPressureGradientFvPatchScalarField onto
        //  a new patch
        timeVaryingPressureGradientFvPatchScalarField
        (
            const timeVaryingPressureGradientFvPatchScalarField&,
            const fvPatch&,
            const DimensionedField<scalar, volMesh>&,
            const fvPatchFieldMapper&
        );

        //- Construct as copy
        timeVaryingPressureGradientFvPatchScalarField
        (
            const timeVaryingPressureGradientFvPatchScalarField&
        ) = delete;

        // //- Construct and return a clone
        // virtual tmp<fvPatchScalarField> clone() const
        // {
        //     return tmp<fvPatchScalarField>
        //     (
        //         new timeVaryingPressureGradientFvPatchScalarField(*this)
        //     );
        // }

        //- Construct as copy setting internal field reference
        timeVaryingPressureGradientFvPatchScalarField
        (
            const timeVaryingPressureGradientFvPatchScalarField&,
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
                new timeVaryingPressureGradientFvPatchScalarField(*this, iF)
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
