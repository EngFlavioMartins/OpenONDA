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

\*---------------------------------------------------------------------------*/

#include "fluxControlledPressureFvPatchScalarField.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::fluxControlledPressureFvPatchScalarField::fluxControlledPressureFvPatchScalarField
(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF
)
:
    fixedGradientFvPatchScalarField(p, iF),
    curTimeIndex_(-1)
{}


Foam::fluxControlledPressureFvPatchScalarField::fluxControlledPressureFvPatchScalarField
(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const dictionary& dict
)
:
    fixedGradientFvPatchScalarField(p, iF, dict, IOobjectOption::MUST_READ),
    curTimeIndex_(-1)
{
    if (dict.found("value") && dict.found("gradient"))
    {
        fvPatchField<scalar>::operator=(scalarField("value", dict, p.size()));
        gradient() = scalarField("gradient", dict, p.size());
    }
    else
    {
        fvPatchField<scalar>::operator=(patchInternalField());
        gradient() = Zero;
    }
}


Foam::fluxControlledPressureFvPatchScalarField::fluxControlledPressureFvPatchScalarField
(
    const fluxControlledPressureFvPatchScalarField& ptf,
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedGradientFvPatchScalarField(ptf, p, iF, mapper),
    curTimeIndex_(-1)
{
    //TODO : Check why we commented this
    // patchType() = ptf.patchType();

    // // Map gradient. Set unmapped values and overwrite with mapped ptf
    // gradient() = 0.0;
    // gradient().map(ptf.gradient(), mapper);

    // // Evaluate the value field from the gradient if the internal field is valid
    // if (notNull(iF) && iF.size())
    // {
    //     scalarField::operator=
    //     (
    //         //patchInternalField() + gradient()/patch().deltaCoeffs()
    //         // ***HGW Hack to avoid the construction of mesh.deltaCoeffs
    //         // which fails for AMI patches for some mapping operations
    //         patchInternalField() + gradient()*(patch().nf() & patch().delta())
    //     );
    // }
    // else
    // {
    //     // Enforce mapping of values so we have a valid starting value. This
    //     // constructor is used when reconstructing fields
    //     this->map(ptf, mapper);
    // }
}

// Foam::fluxControlledPressureFvPatchScalarField::fluxControlledPressureFvPatchScalarField
// (
//     const fluxControlledPressureFvPatchScalarField& wbppsf
    
// )
// :
//     fixedGradientFvPatchScalarField(wbppsf),
//     curTimeIndex_(-1)
// {}


Foam::fluxControlledPressureFvPatchScalarField::fluxControlledPressureFvPatchScalarField
(
    const fluxControlledPressureFvPatchScalarField& wbppsf,
    const DimensionedField<scalar, volMesh>& iF
)
:
    fixedGradientFvPatchScalarField(wbppsf, iF),
    curTimeIndex_(-1)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::fluxControlledPressureFvPatchScalarField::updateCoeffs
(
    const scalarField& snGradp
)
{
    if (updated())
    {
        return;
    }

    curTimeIndex_ = this->db().time().timeIndex();

    gradient() = snGradp;
    fixedGradientFvPatchScalarField::updateCoeffs();
}


void Foam::fluxControlledPressureFvPatchScalarField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    if (curTimeIndex_ != this->db().time().timeIndex())
    {
        FatalErrorInFunction
            << "updateCoeffs(const scalarField& snGradp) MUST be called before"
               " updateCoeffs() or evaluate() to set the boundary gradient."
            << exit(FatalError);
    }
}


void Foam::fluxControlledPressureFvPatchScalarField::write(Ostream& os) const
{
    fixedGradientFvPatchScalarField::write(os);
    writeEntry("value", os);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    makePatchTypeField
    (
        fvPatchScalarField,
        fluxControlledPressureFvPatchScalarField
    );
}


// ************************************************************************* //
