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

#include "timeVaryingOutletInletFvPatchField.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class Type>
Foam::timeVaryingOutletInletFvPatchField<Type>::timeVaryingOutletInletFvPatchField
(
    const fvPatch& p,
    const DimensionedField<Type, volMesh>& iF
)
:
    mixedFvPatchField<Type>(p, iF),
    phiName_("phi")
{
    this->refValue() = *this;
    this->refGrad() = Zero;
    this->valueFraction() = 0.0;
}



template<class Type>
Foam::timeVaryingOutletInletFvPatchField<Type>::timeVaryingOutletInletFvPatchField
(
    const fvPatch& p,
    const DimensionedField<Type, volMesh>& iF,
    const dictionary& dict
)
:
    mixedFvPatchField<Type>(p, iF),
    phiName_(dict.lookupOrDefault<word>("phi", "phi"))
{
    this->refValue() = Field<Type>("outletValue", dict, p.size());
    this->refGrad() = Field<Type>("inletGradient", dict, p.size());

    if (dict.found("value"))
    {
        fvPatchField<Type>::operator=
        (
            Field<Type>("value", dict, p.size())
        );
    }
    else
    {
        fvPatchField<Type>::operator=(this->refValue());
    }

    this->valueFraction() = 0.0;
}

template<class Type>
Foam::timeVaryingOutletInletFvPatchField<Type>::timeVaryingOutletInletFvPatchField
(
    const timeVaryingOutletInletFvPatchField<Type>& ptf,
    const fvPatch& p,
    const DimensionedField<Type, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    mixedFvPatchField<Type>(ptf, p, iF, mapper),
    phiName_(ptf.phiName_)
{}

// template<class Type>
// Foam::timeVaryingOutletInletFvPatchField<Type>::timeVaryingOutletInletFvPatchField
// (
//     const timeVaryingOutletInletFvPatchField<Type>& ptf
// )
// :
//     mixedFvPatchField<Type>(ptf),
//     phiName_(ptf.phiName_)
// {}


template<class Type>
Foam::timeVaryingOutletInletFvPatchField<Type>::timeVaryingOutletInletFvPatchField
(
    const timeVaryingOutletInletFvPatchField<Type>& ptf,
    const DimensionedField<Type, volMesh>& iF
)
:
    mixedFvPatchField<Type>(ptf, iF),
    phiName_(ptf.phiName_)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class Type>
void Foam::timeVaryingOutletInletFvPatchField<Type>::updateCoeffs()
{
    if (this->updated())
    {
        return;
    }

    const Field<scalar>& phip =
        this->patch().template lookupPatchField<surfaceScalarField, scalar>
        (
            phiName_
        );

    this->valueFraction() = pos(phip);

    mixedFvPatchField<Type>::updateCoeffs();
}

template<class Type>
void Foam::timeVaryingOutletInletFvPatchField<Type>::updateRefValue(const Field<Type>& newRefValue)
{
    this->refValue() = newRefValue;
    timeVaryingOutletInletFvPatchField<Type>::updateCoeffs();
}

template<class Type>
void Foam::timeVaryingOutletInletFvPatchField<Type>::updateRefGradient(const Field<Type>& newRefGradient)
{
    this->refGrad() = newRefGradient;
    timeVaryingOutletInletFvPatchField<Type>::updateCoeffs();
}

template<class Type>
void Foam::timeVaryingOutletInletFvPatchField<Type>::write(Ostream& os) const
{
    fvPatchField<Type>::write(os);
    if (phiName_ != "phi")
    {
        this->writeEntry("phi", os);  // Corrected function call
    }
    this->writeEntry("inletValue", os);  // Corrected function call
    this->writeEntry("value", os);  // Corrected function call
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    makePatchTypeFieldTypedefs(timeVaryingOutletInlet);
    makePatchFields(timeVaryingOutletInlet);
}

// ************************************************************************* //
