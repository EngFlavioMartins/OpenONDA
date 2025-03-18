/*---------------------------------------------------------------------------*\
=========                 |
\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
\\    /   O peration     |
\\  /    A nd           | Copyright (C) 2011-2015 OpenFOAM Foundation
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

Application
pimpleFoam

Description
Large time-step transient solver for incompressible, flow using the PIMPLE
(merged PISO-SIMPLE) algorithm.

Sub-models include:
- turbulence modelling, i.e. laminar, RAS or LES
- run-time selectable MRF and finite volume options, e.g. explicit porosity

\*---------------------------------------------------------------------------*/


#include "foamSolverCore.H"

// ==================================================
// Constructors and Destructor
// ==================================================
cppFoamSolver::cppFoamSolver(int ARGC, char *ARGV[])
{

int _argc = ARGC;
char** _argv = ARGV;

argc = _argc;
argv = _argv;

_args = new Foam::argList(argc, argv);
if (!_args->checkRootCase())
{
      Foam::FatalError.exit();
}
Foam::argList& args = *_args;

//-------------------------------------------------------------------------
// Origin of attribute: pimpleFoam.C/createTime.H
Foam::Info<< "Create time\n" << Foam::endl;

// Foam::Time _runTime(Foam::Time::controlDictName, _args);
_runTime = new Foam::Time(Foam::Time::controlDictName, args);
Foam::Time& runTime = *_runTime;

//-------------------------------------------------------------------------
// Origin of attribute: pimpleFoam.C/createDynamicFvMesh.H
// TODO: Check if this is correct
Foam::Info<< "Create mesh for time = " << runTime.timeName() << Foam::endl;

_meshPtr = autoPtr<dynamicFvMesh> 
(
      dynamicFvMesh::New
      (
            IOobject
            (
            dynamicFvMesh::defaultRegion,
            runTime.timeName(),
            runTime,
            IOobject::MUST_READ
            )
      )
);

dynamicFvMesh& _mesh = _meshPtr();
meshPtr = &_mesh;

//-------------------------------------------------------------------------
// Origin of attribute: pimpleFoam.C/initContinuityErrs.H
cumulativeContErr = 0;

//-------------------------------------------------------------------------
// Origin of attribute: /usr/lib/openfoam/openfoam2406/src/dynamicFvMesh/include
#if defined(NO_CONTROL)

#elif defined(PIMPLE_CONTROL)
_pimple = new pimpleControl(_mesh);
pimpleControl& pimple = *_pimple;

#endif

//-------------------------------------------------------------------------
// Origin of attribute: /usr/lib/openfoam/openfoam2406/src/finiteVolume/lnInclude/createTimeControls.H

bool adjustTimeStep =
    runTime.controlDict().getOrDefault("adjustTimeStep", false);

scalar maxCo = runTime.controlDict().getOrDefault<scalar>("maxCo", 1);

scalar maxDeltaT =runTime.controlDict().getOrDefault<scalar>("maxDeltaT", GREAT);


//-------------------------------------------------------------------------

_correctPhi = new bool(pimple.dict().lookupOrDefault("correctPhi", _mesh.dynamic()));
bool& correctPhi = *_correctPhi;

_checkMeshCourantNo = new bool(pimple.dict().lookupOrDefault("moveMeshOuterCorrectors", false));
bool& checkMeshCourantNo = *_checkMeshCourantNo;

_moveMeshOuterCorrectors = new bool(pimple.dict().lookupOrDefault("moveMeshOuterCorrectors", false));
bool& moveMeshOuterCorrectors = *_moveMeshOuterCorrectors;

//-------------------------------------------------------------------------
// Origin of attribute: pimpleFoam.C/createFields.H


// Origin of attribute: pimpleFoam.C/createRDeltaT.H
LTS = fv::localEulerDdt::enabled(_mesh);

if (LTS){
      Info<< "Using LTS" << endl;

      trDeltaT = tmp<volScalarField>
      (
            new volScalarField
            (
            IOobject
            (
                  fv::localEulerDdt::rDeltaTName,
                  runTime.timeName(),
                  _mesh,
                  IOobject::READ_IF_PRESENT,
                  IOobject::AUTO_WRITE
            ),
            _mesh,
            dimensionedScalar(dimless/dimTime, 1),
            extrapolatedCalculatedFvPatchScalarField::typeName
            )
      );
}

Info<< "Reading field p\n" << endl;

_p = new volScalarField
(
      IOobject
      (
            "p",
            runTime.timeName(),
            _mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
      ),
      _mesh
);
volScalarField& p = *_p;


Info<< "Reading field U\n" << endl;
_U = new volVectorField 
(
      IOobject
      (
            "U",
            runTime.timeName(),
            _mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
      ),
      _mesh
);
volVectorField& U = *_U;

//-------------------------------------------------------------------------
// Origin of attribute: icoFoam.C/createFields.H

_transportProperties = new IOdictionary
(
      IOobject
      (
            "transportProperties",
            runTime.constant(),
            _mesh,
            IOobject::MUST_READ_IF_MODIFIED,
            IOobject::NO_WRITE
      )
);
IOdictionary& transportProperties = *_transportProperties;

_nu = new dimensionedScalar
(
    transportProperties.lookupOrDefault<dimensionedScalar>("nu", dimensionedScalar(dimViscosity, 1.5e-12))
);

dimensionedScalar& nu = *_nu;


//-------------------------------------------------------------------------
// Origin of attribute: pimpleFoam.C/createFields.H/createPhi.H

Info<< "Reading/calculating face flux field phi\n" << endl;

_phi = new surfaceScalarField
(
      IOobject
      (
            "phi",
            runTime.timeName(),
            _mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::AUTO_WRITE
      ),
      fvc::flux(U)
);
surfaceScalarField& phi = *_phi;

//-------------------------------------------------------------------------
// Origin of attribute: pxxxx

_mesh.setFluxRequired(p.name());

_laminarTransport = new singlePhaseTransportModel(U, phi);
singlePhaseTransportModel& laminarTransport = *_laminarTransport;


_turbulence = new autoPtr<incompressible::turbulenceModel>
(
      incompressible::turbulenceModel::New(U, phi, laminarTransport)
);

autoPtr<incompressible::turbulenceModel>& turbulence = *_turbulence;

//-------------------------------------------------------------------------
// Origin of attribute: xxxx

_MRF = new IOMRFZoneList(_mesh);
IOMRFZoneList& MRF = *_MRF;

//-------------------------------------------------------------------------
// Origin of attribute: /usr/lib/openfoam/openfoam2406/src/finiteVolume/cfdTools/incompressible/createUfIfPresent.H

if (_mesh.dynamic())
{
      Info<< "Constructing face velocity Uf\n" << endl;

      Uf = autoPtr<surfaceVectorField>
      (
            new surfaceVectorField
            (
                  IOobject
                  (
                        "Uf",
                        runTime.timeName(),
                        _mesh,
                        IOobject::READ_IF_PRESENT,
                        IOobject::AUTO_WRITE
                  ),
                  fvc::interpolate(U)
            )
      );

}


//-------------------------------------------------------------------------
// Origin of attribute: pimpleFoam.C

turbulence->validate();

//-------------------------------------------------------------------------
// Origin of attribute: /usr/lib/openfoam/openfoam2406/src/finiteVolume/cfdTools/incompressible/CourantNo.H

if (!LTS)
{
      scalar CoNum = 0.0;
      scalar meanCoNum = 0.0;
      
      {
          scalarField sumPhi
          (
              fvc::surfaceSum(mag(phi))().primitiveField()
          );
      
          CoNum = 0.5*gMax(sumPhi/_mesh.V().field())*runTime.deltaTValue();
      
          meanCoNum =
              0.5*(gSum(sumPhi)/gSum(_mesh.V().field()))*runTime.deltaTValue();
      }
      
      Info<< "Courant Number mean: " << meanCoNum
          << " max: " << CoNum << endl;
      
      #include "setInitialDeltaT.H"
}

// -------------------------------------------------------------------------
// Custom attributes

_vorticity = new volVectorField
(
      IOobject
      (
            "vorticity",
            runTime.timeName(),
            _mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
      ),
      fvc::curl(U)
);
volVectorField& vorticity = *_vorticity;
vorticity.write();


// Check if the numerical boundary was implemented
int numBoundLabel = _mesh.boundaryMesh().findPatchID("numericalBoundary");

if (numBoundLabel == -1)
{
    FatalErrorInFunction
        << "Patch 'numericalBoundary' not found in mesh boundary!"
        << exit(FatalError);
}
else
{
    Info << "Patch 'numericalBoundary' exists with index: " << numBoundLabel << endl;
}

_inletRefValue = new vectorField(0 * _mesh.boundaryMesh()[numBoundLabel].faceAreas());

// inletRefValuePtr = &_inletRefValue;
vectorField& inletRefValue = *_inletRefValue;

_inletRefGradient = new  vectorField(0 * _mesh.boundaryMesh()[numBoundLabel].faceAreas());
// inletRefGradientPtr = &_inletRefGradient;
vectorField& inletRefGradient = *_inletRefGradient;

_outletRefGradient = new vectorField(0 * _mesh.boundaryMesh()[numBoundLabel].faceAreas());
// outletRefGradientPtr = &_outletRefGradient;
vectorField& outletRefGradient = *_outletRefGradient;

_outletRefValue = new scalarField(0 * mag(_mesh.boundaryMesh()[numBoundLabel].faceAreas()));
// outletRefValuePtr = &_outletRefValue;
scalarField& outletRefValue = *_outletRefValue;

}

// ==================================================
// Destructor
// ==================================================

cppFoamSolver::~cppFoamSolver(){
      Info << "C++ Object destroyed " << endl;
}

// ==================================================
// Simulation Methods
// ==================================================

// checkPatch method that verifies if a patch exists in the mesh boundary. If the patch is not found, it prints an error and lists all available patches.
void cppFoamSolver::checkPatch(char *patchName)
{
      dynamicFvMesh& mesh= *meshPtr;
      int patchIndex = mesh.boundaryMesh().findPatchID(patchName);

      if (patchIndex == -1)
      {
            FatalErrorInFunction
                  << "Error: Patch '" << patchName << "' not found in the mesh boundary!" << nl
                  << "Available patches: " << nl;

            forAll(mesh.boundaryMesh(), patchI)
            {
                  Info << " - " << mesh.boundaryMesh()[patchI].name() << nl;
            }

            FatalError.exit();
      }
}

void cppFoamSolver::evolve()
{
      Foam::argList& args = *_args;
      Foam::Time& runTime = *_runTime;
      dynamicFvMesh& mesh= *meshPtr;

      pimpleControl& pimple = *_pimple;    
      bool& correctPhi = *_correctPhi;
      bool& checkMeshCourantNo = *_checkMeshCourantNo;
      bool& moveMeshOuterCorrectors = *_moveMeshOuterCorrectors;

      volScalarField& p = *_p;
      volVectorField& vorticity = *_vorticity;
      volVectorField& U = *_U;
      surfaceScalarField& phi = *_phi;

      dimensionedScalar& nu = *_nu;
      IOdictionary& transportProperties = *_transportProperties;
      

      label pRefCell = 0;
      scalar pRefValue = 0.0;
      setRefCell(p, pimple.dict(), pRefCell, pRefValue);
      mesh.setFluxRequired(p.name());

      singlePhaseTransportModel& laminarTransport = *_laminarTransport;
      autoPtr<incompressible::turbulenceModel>& turbulence = *_turbulence;
      IOMRFZoneList& MRF = *_MRF;
      
      fv::options fvOptions(mesh);

      //TODO: Check this
      scalar& CoNum = *CoNumPtr;
      scalar& meanCoNum = *meanCoNumPtr;

      scalarField& sumPhi = *sumPhiPtr;
      dimensionedScalar dt = runTime.deltaT();

      vectorField& inletRefValue = *_inletRefValue;
      vectorField& inletRefGradient = *_inletRefGradient;
      vectorField& outletRefGradient = *_outletRefGradient;
      scalarField& outletRefValue = *_outletRefValue;

      runTime.run();

      Info<< "\n cppFoamSolver:: Updating solution\n" << endl;
      
      #include "readDyMControls.H"

      if (LTS)
      {
            #include "setRDeltaT.H"
      }
      else
      {
            #include "CourantNo.H"
            #include "setDeltaT.H"
      }

      ++runTime;

      Info<< "Time = " << runTime.timeName() << nl << endl;

      volVectorField::Boundary& UBf = U.boundaryFieldRef();
      volScalarField::Boundary& pBf = p.boundaryFieldRef();

      #include "updateInletValue.H"
      #include "updateOutletGradient.H"
      #include "updateInletGradient.H"
      #include "updateOutletValue.H"


      // --- Pressure-velocity PIMPLE corrector loop
      while (pimple.loop())
      {
            if (pimple.firstIter() || moveMeshOuterCorrectors)
            {
                  // Do any mesh changes
                  mesh.controlledUpdate();

                  if (mesh.changing())
                  {
                        MRF.update();

                        if (correctPhi)
                        {     
                              // Calculate absolute flux
                              // from the mapped surface velocity
                              phi = mesh.Sf() & Uf();


                              #include "correctPhi.H"

                              // Make the flux relative to the mesh motion
                              fvc::makeRelative(phi, U);

                        }

                        if (checkMeshCourantNo)
                        {
                              #include "meshCourantNo.H"
                        }
                  }
            }
            
            #include "UEqn.H"

            // --- Pressure corrector loop
            while (pimple.correct())
            {     
                  #include "pEqn.H"
            }

            if (pimple.turbCorr())
            {
                  laminarTransport.correct();
                  turbulence->correct();
            }
      }

      vorticity = fvc::curl(U);

      runTime.write();

      runTime.printExecutionTime(Info);

}



void cppFoamSolver::evolve_mesh(){
      Foam::argList& args = *_args;
      Foam::Time& runTime = *_runTime;
      dynamicFvMesh& mesh= *meshPtr;

      pimpleControl& pimple = *_pimple;    
      bool& correctPhi = *_correctPhi;
      bool& checkMeshCourantNo = *_checkMeshCourantNo;
      bool& moveMeshOuterCorrectors = *_moveMeshOuterCorrectors;
      volScalarField& p = *_p;
      volVectorField& U = *_U;
      dimensionedScalar& nu = *_nu;
      IOdictionary& transportProperties = *_transportProperties;
      surfaceScalarField& phi = *_phi;
      singlePhaseTransportModel& laminarTransport = *_laminarTransport;
      autoPtr<incompressible::turbulenceModel>& turbulence = *_turbulence;
      IOMRFZoneList& MRF = *_MRF;
      volVectorField& vorticity = *_vorticity;
      fv::options fvOptions(mesh);
      label pRefCell = 0;
      scalar pRefValue = 0.0;
      setRefCell(p, pimple.dict(), pRefCell, pRefValue);
      mesh.setFluxRequired(p.name());


      //TODO: Check this
      scalar& CoNum = *CoNumPtr;
      scalar& meanCoNum = *meanCoNumPtr;
      scalarField& sumPhi = *sumPhiPtr;
      dimensionedScalar dt = runTime.deltaT();
      vectorField& inletRefValue = *_inletRefValue;
      vectorField& inletRefGradient = *_inletRefGradient;
      vectorField& outletRefGradient = *_outletRefGradient;
      scalarField& outletRefValue = *_outletRefValue;

      Info<< "\n cppFoamSolver:: Updating mesh\n" << endl;

      #include "readDyMControls.H"

      if (LTS)
      {
            #include "setRDeltaT.H"
      }
      else
      {
            #include "CourantNo.H"
            #include "setDeltaT.H"
      }

      ++runTime;

      mesh.controlledUpdate();

      if (mesh.changing())
      {
            MRF.update();

            if (correctPhi)
            {
                  // Calculate absolute flux
                  // from the mapped surface velocity
                  phi = mesh.Sf() & Uf();

                  #include "correctPhi.H"

                  // Make the flux relative to the mesh motion
                  fvc::makeRelative(phi, U);
            }

            if (checkMeshCourantNo)
            {
                  #include "meshCourantNo.H"
            }
      }
}


void cppFoamSolver::evolve_only_solution(){
      Foam::argList& args = *_args;
      Foam::Time& runTime = *_runTime;
      dynamicFvMesh& mesh= *meshPtr;

      pimpleControl& pimple = *_pimple;    
      bool& correctPhi = *_correctPhi;
      bool& checkMeshCourantNo = *_checkMeshCourantNo;
      bool& moveMeshOuterCorrectors = *_moveMeshOuterCorrectors;

      volScalarField& p = *_p;
      volVectorField& vorticity = *_vorticity;
      volVectorField& U = *_U;
      surfaceScalarField& phi = *_phi;

      dimensionedScalar& nu = *_nu;
      IOdictionary& transportProperties = *_transportProperties;
      

      label pRefCell = 0;
      scalar pRefValue = 0.0;
      setRefCell(p, pimple.dict(), pRefCell, pRefValue);
      mesh.setFluxRequired(p.name());

      singlePhaseTransportModel& laminarTransport = *_laminarTransport;
      autoPtr<incompressible::turbulenceModel>& turbulence = *_turbulence;
      IOMRFZoneList& MRF = *_MRF;
      
      fv::options fvOptions(mesh);

      //TODO: Check this
      scalar& CoNum = *CoNumPtr;
      scalar& meanCoNum = *meanCoNumPtr;

      scalarField& sumPhi = *sumPhiPtr;
      dimensionedScalar dt = runTime.deltaT();

      vectorField& inletRefValue = *_inletRefValue;
      vectorField& inletRefGradient = *_inletRefGradient;
      vectorField& outletRefGradient = *_outletRefGradient;
      scalarField& outletRefValue = *_outletRefValue;

      runTime.run();

      Info<< "\n cppFoamSolver:: Updating solution only\n" << endl;
      
      #include "readDyMControls.H"

      if (LTS)
      {
            #include "setRDeltaT.H"
      }
      else
      {
            #include "CourantNo.H"
            #include "setDeltaT.H"
      }

      ++runTime;

      Info<< "Time = " << runTime.timeName() << nl << endl;

      volVectorField::Boundary& UBf = U.boundaryFieldRef();
      volScalarField::Boundary& pBf = p.boundaryFieldRef();

      #include "updateInletValue.H"
      #include "updateOutletGradient.H"
      #include "updateInletGradient.H"
      #include "updateOutletValue.H"


      // --- Pressure-velocity PIMPLE corrector loop
      while (pimple.loop())
      {
            #include "UEqn.H"

            // --- Pressure corrector loop
            while (pimple.correct())
            {     
                  #include "pEqn.H"
            }

            if (pimple.turbCorr())
            {
                  laminarTransport.correct();
                  turbulence->correct();
            }
      }

      vorticity = fvc::curl(U);

      runTime.write();

      runTime.printExecutionTime(Info);
}



void cppFoamSolver::correct_mass_flux(char *patchName){

      checkPatch(patchName);
      dynamicFvMesh& mesh= *meshPtr;
      volVectorField& U = *_U;

      // Create polyPatches for the numerical boundary
      const polyPatch& nbPatch = mesh.boundaryMesh()[boundary_label(patchName)];

      // Extract face normal vectors
      const Field<vector>& fn  = nbPatch.faceNormals();

      
      // Extract face areas
      const Field<vector>& fa = nbPatch.faceAreas();

      Field<scalar> sPhi(fn.size());
      Field<scalar> cPhi(fn.size());

      // Compute cos(phi) and sin(phi) for each face, where phi is the angle
      // between the face normal and [0 1 0]^T
      for(int k = 0; k<fn.size();k++){
            sPhi[k] = -fn[k][0]/mag(fn[k]);
            cPhi[k] = fn[k][1]/mag(fn[k]);
      }

      float Qnet = 0;
      float Qtotal = 0;
      float tmp = 0.0;


      // Compute the net and total volumetric flow rates
      for(int k = 0; k < U.boundaryFieldRef()[boundary_label(patchName)].size(); k++){
            tmp = fa[k]&U.boundaryFieldRef()[boundary_label(patchName)][k];
            Qnet   += tmp;
            Qtotal += mag(tmp);
      }

      Tensor<scalar> R(1, 0, 0, 0, 1, 0, 0, 0, 1);

      // Rotate boundary velocity to cell-face reference frame
      // Apply flux correction
      // Rotate boundary velocity back to Cartesian reference frame
      vectorField& Ub = U.boundaryFieldRef()[boundary_label(patchName)];

      Info << "Total volumetric flow rate through the boundary patch: " << Qtotal << " m3/s" << endl;

      if (mag(Qtotal) > SMALL) // Prevent division by zero, in OpenFOAM: 1e-12
      {
          for(int k = 0; k < fn.size(); k++)
          {
            // Construct the rotation matrix
            R.xx() = cPhi[k];
            R.xy() = sPhi[k];
            R.yx() = -sPhi[k];
            R.yy() = cPhi[k];

            // Rotate to local frame
            Ub[k] = transform(R, Ub[k]);

            // Apply correction in the local frame
            Ub[k][1] -= (Qnet / Qtotal) * mag(Ub[k][1]);

            // Rotate back to Cartesian reference frame
            Ub[k] = transform(R.T(), Ub[k]);

            Info << "Before correction: " << Ub[k] << endl;
            Info << "Qnet: " << Qnet << " Qtotal: " << Qtotal << endl;
            Info << "Rotation matrix: " << R << endl;
            Info << "After correction: " << Ub[k] << endl;
          }
      }
}

// ==================================================
// Get Methods (Field and Mesh Data)
// ==================================================
double cppFoamSolver::get_run_time_value(){
      Foam::Time& runTime = *_runTime;
      return runTime.value();
}

double cppFoamSolver::get_time_step(){
      Foam::Time& runTime = *_runTime;
      return runTime.deltaT().value();
}

int cppFoamSolver::get_number_of_nodes(){
      dynamicFvMesh& mesh= *meshPtr;
      return mesh.nPoints();
}

int cppFoamSolver::get_number_of_cells(){
      dynamicFvMesh& mesh= *meshPtr;
      return mesh.nCells();
}

int cppFoamSolver::get_number_of_boundary_nodes(char *patchName){
      checkPatch(patchName);
      dynamicFvMesh& mesh= *meshPtr;
      return mesh.boundaryMesh()[boundary_label(patchName)].nPoints();
}

int cppFoamSolver::get_number_of_boundary_faces(char *patchName){
      checkPatch(patchName);
      dynamicFvMesh& mesh= *meshPtr;
      return mesh.boundaryMesh()[boundary_label(patchName)].size();
}


// Cell and Node Coordinate Access
void cppFoamSolver::get_node_coordinates(double *coordinates){
      dynamicFvMesh& mesh= *meshPtr;
      int index = 0;

      // Copy node coordinates of the entire mesh to C-style array
      forAll( mesh.points(), nodeI ){
            for ( int dim = 0; dim < 3; dim++ ){
                  coordinates[index] = mesh.points()[nodeI].component(dim);
                  index++;
            }
      }
}

void cppFoamSolver::get_connectivity(int *connectivity){
      dynamicFvMesh& mesh= *meshPtr;
      int index = 0;

      forAll( mesh.cellPoints(), cellI ){
            for ( int j = 0; j < 8; j++ ){
                  connectivity[index] = mesh.cellPoints()[cellI][j];
                  index++;
            }
      }
}

void cppFoamSolver::get_cell_volumes(double *volumes){
      dynamicFvMesh& mesh= *meshPtr;
      int index = 0;

      forAll( mesh.V(), cellI){
            volumes[index] = mesh.V()[cellI];
            index++;
      }
}

void cppFoamSolver::get_cell_center_coordinates(double *coordinates){
      dynamicFvMesh& mesh= *meshPtr;
      int index = 0;

      // Copy cell-center coordinates of internal field to C-style array
      forAll( mesh.C(), cellI ){
            for ( int dim = 0; dim < 3; dim++ ){
                  coordinates[index] = mesh.C()[cellI].component(dim);
                  index++;
            }
      }
}


// Boundary Data Access
void cppFoamSolver::get_boundary_node_coordinates(double *coordinates, char *patchName){
      checkPatch(patchName);
      dynamicFvMesh& mesh= *meshPtr;
      int index = 0;
      const vectorField boundaryPoints = mesh.boundaryMesh()[boundary_label(patchName)].localPoints();

      // Copy node coordinates of nodes on the numerical boundary to
      // C-style array
      forAll( boundaryPoints, nodeI ){
            coordinates[index]   = boundaryPoints[nodeI].x();
            coordinates[index+1] = boundaryPoints[nodeI].y();
            coordinates[index+2] = boundaryPoints[nodeI].z();
            index += 3;
      }
}

void cppFoamSolver::get_boundary_node_normal(double *normals, char *patchName){
      checkPatch(patchName);
      dynamicFvMesh& mesh= *meshPtr;
      const vectorField boundaryPointNormals = mesh.boundaryMesh()[boundary_label(patchName)].pointNormals();
      int index = 0;

      forAll( boundaryPointNormals, pointI ){
            normals[index]   = boundaryPointNormals[pointI].x();
            normals[index+1] = boundaryPointNormals[pointI].y();
            normals[index+2] = boundaryPointNormals[pointI].z();
            index += 3;
      }
}

void cppFoamSolver::get_boundary_face_center_coordinates(double *coordinates, char *patchName){
      checkPatch(patchName);
      dynamicFvMesh& mesh= *meshPtr;
      int index = 0;

      // Copy face-center coordinates of faces on the numerical boundary to
      // C-style array
      forAll( mesh.boundaryMesh()[boundary_label(patchName)].faceCentres(), faceI ){
            coordinates[index]   = mesh.boundaryMesh()[boundary_label(patchName)].faceCentres()
                                    [faceI].x();
            coordinates[index+1] = mesh.boundaryMesh()[boundary_label(patchName)].faceCentres()
                                    [faceI].y();
            coordinates[index+2] = mesh.boundaryMesh()[boundary_label(patchName)].faceCentres()
                                    [faceI].z();
            index += 3;
      }
}

void cppFoamSolver::get_boundary_face_areas(double *areas, char *patchName){
      checkPatch(patchName);
      dynamicFvMesh& mesh= *meshPtr;
      int index = 0;

      forAll( mesh.boundaryMesh()[boundary_label(patchName)].faceAreas(), faceI ){
            areas[index] = mag(mesh.boundaryMesh()[boundary_label(patchName)].faceAreas()[faceI]);
            index++;
      }
}

void cppFoamSolver::get_boundary_face_normals(double *normals, char *patchName){
      checkPatch(patchName);
      dynamicFvMesh& mesh= *meshPtr;
      int index = 0;

      forAll( mesh.boundaryMesh()[boundary_label(patchName)].faceNormals(), faceI ){
            normals[index]   = mesh.boundaryMesh()[boundary_label(patchName)].faceNormals()[faceI].x();
            normals[index+1] = mesh.boundaryMesh()[boundary_label(patchName)].faceNormals()[faceI].y();
            normals[index+2] = mesh.boundaryMesh()[boundary_label(patchName)].faceNormals()[faceI].z();
            index += 3;
      }
}

void cppFoamSolver::get_boundary_cell_center_coordinates(double *coordinates, char *patchName){
      checkPatch(patchName);
      Foam::fvMesh& mesh = *meshPtr;
      volVectorField cellCenter = mesh.C();
      int index = 0;
      int cellID = 0;

      // Copy face-cell-center coordinates of faces on the numerical boundary to
      // C-style array
      forAll( mesh.boundaryMesh()[boundary_label(patchName)].faceCells(), cellI ){
            cellID = mesh.boundaryMesh()[boundary_label(patchName)].faceCells()[cellI];
            coordinates[index]   = cellCenter[cellID].x();
            coordinates[index+1] = cellCenter[cellID].y();
            coordinates[index+2] = cellCenter[cellID].z();
            index += 3;
      }
}


// Field Access Methods
void cppFoamSolver::get_velocity_field(double *pyVelocity){
      // volVectorField& U = *UPtr;
      volVectorField& U = *_U;
      int index = 0;

      forAll( U.internalField(), cellI ){
            for ( int dim = 0; dim < 3; dim++ ){
                  pyVelocity[index] = U.internalField()[cellI][dim];
                  index++;
            }
      }
}

void cppFoamSolver::get_velocity_boundary_field(double *pyVelocity, char *patchName){
      checkPatch(patchName);
      volVectorField& U = *_U;
      int index = 0;

      forAll( U.boundaryFieldRef()[boundary_label(patchName)], cellI ){
            for ( int dim = 0; dim < 3; dim++ ){
                  pyVelocity[index] = U.boundaryFieldRef()[boundary_label(patchName)][cellI][dim];
                  index++;
            }
      }
}

void cppFoamSolver::get_pressure_field(double *pyPressure){
      volScalarField& p = *_p;

      forAll( p.internalField(), cellI ){
            pyPressure[cellI] = p.internalField()[cellI];
      }
}

void cppFoamSolver::get_velocity_gradient(double *pyVelocityGradient){
      volVectorField& U = *_U;
      volTensorField dU = fvc::grad(U);

      int index = 0;

      forAll( dU.internalField(), cellI ){
            for ( int dim = 0; dim < 9; dim++ ){
                  pyVelocityGradient[index] = dU.internalField()[cellI][dim];
                  index++;
            }
      }
}

void cppFoamSolver::get_velocity_gradient_boundary_field(double *pyVelocityGradient, char *patchName){
      checkPatch(patchName);
      volVectorField& U = *_U;
      volTensorField dU = fvc::grad(U);

      int index = 0;

      // Copy velocity gradient components stored in volVectorField obdimect to C-style
      forAll( dU.boundaryFieldRef()[boundary_label(patchName)], cellI ){
            for ( int dim = 0; dim < 9; dim++ ){
                  pyVelocityGradient[index] = dU.boundaryFieldRef()[boundary_label(patchName)][cellI][dim];
                  index++;
            }
      }
}

void cppFoamSolver::get_pressure_gradient_field(double *pyPressureGradient)
{
      volScalarField& p = *_p;
      volVectorField dp = fvc::grad(p);
      int index = 0;

      // Copy pressure gradient components stored in volVectorField object to C-style
      forAll( dp.internalField(), cellI ){
            for ( int dim = 0; dim < 3; dim++ ){
                  pyPressureGradient[index] = dp.internalField()[cellI][dim];
                  index++;
            }
      }
}


void cppFoamSolver::get_pressure_boundary_field(double *pyPressure, char *patchName){
      checkPatch(patchName);
      volScalarField& p = *_p;

      forAll( p.boundaryFieldRef()[boundary_label(patchName)], cellI ){
            pyPressure[cellI] = p.boundaryFieldRef()[boundary_label(patchName)][cellI];
      }
}

void cppFoamSolver::get_pressure_gradient_boundary_field(double *pyPressureGradient, char *patchName){
      checkPatch(patchName);
      volScalarField& p = *_p;
      volVectorField dp = fvc::grad(p);
      int index = 0;

      // Copy pressure gradient components stored in volVectorField object to C-style
      // array
      forAll( dp.boundaryFieldRef()[boundary_label(patchName)], cellI ){
            for ( int dim = 0; dim < 3; dim++ ){
                  pyPressureGradient[index] = dp.boundaryFieldRef()[boundary_label(patchName)][cellI][dim];
                  index++;
            }
      }
}

void cppFoamSolver::get_vorticity_field(double *pyVorticity){
      volVectorField& vorticity = *_vorticity;
      int index = 0;

      // Copy vorticity components stored in volVectorField object to C-style
      // array
      forAll( vorticity.internalField(), cellI ){
            for ( int dim = 0; dim < 3; dim++ ){
                  pyVorticity[index] = vorticity.internalField()[cellI][dim];
                  index++;
            }
      }
}

void cppFoamSolver::get_vorticity_boundary_field(double *pyVorticity, char *patchName){
      checkPatch(patchName);
      volVectorField& vorticity = *_vorticity;
      int index = 0;

      // Copy vorticity components stored in volVectorField object to C-style
      // array
      forAll( vorticity.boundaryFieldRef()[boundary_label(patchName)], cellI ){
            for ( int dim = 0; dim < 3; dim++ ){
                  pyVorticity[index] = vorticity.boundaryFieldRef()[boundary_label(patchName)][cellI][dim];
                  index++;
            }
      }
}


// ==================================================
// Set Methods (Boundary and Simulation Data)
// ==================================================
void cppFoamSolver::set_time_step(double *tStep){
      Foam::Time& runTime = *_runTime;
      runTime.setDeltaT(*tStep);
}

void cppFoamSolver::set_dirichlet_velocity_boundary_condition(double *pyVelocityBC, char *patchName){
      checkPatch(patchName);
      volVectorField& U = *_U;

      int index = 0;

      forAll( U.boundaryFieldRef()[boundary_label(patchName)], cellI ){
            for ( int dim = 0; dim < 3; dim++ ){
                  U.boundaryFieldRef()[boundary_label(patchName)][cellI][dim] = pyVelocityBC[index];
                  index++;
            }
      }
}

void cppFoamSolver::set_dirichlet_pressure_boundary_condition(double *pyPressureBC, char *patchName){
      checkPatch(patchName);
      volScalarField& p = *_p;

      forAll( p.boundaryFieldRef()[boundary_label(patchName)], cellI ){
            p.boundaryFieldRef()[boundary_label(patchName)][cellI] = pyPressureBC[cellI];
      }
}

void cppFoamSolver::set_neumann_pressure_boundary_condition(double *pyPressureGradientBC, char *patchName){
      checkPatch(patchName);
      volScalarField& p = *_p;
      volScalarField::Boundary& pBf = p.boundaryFieldRef();
      dynamicFvMesh& mesh= *meshPtr;

      int index = 0;
      int numFaces = get_number_of_boundary_faces(patchName);

      vectorField gradP(numFaces);
      for (int cellI=0; cellI<numFaces; cellI++){
            for (int dim=0; dim<3; dim++){
                  gradP[cellI][dim] = pyPressureGradientBC[index];
                  index++;
            }
      }

      if (isA<timeVaryingPressureGradientFvPatchScalarField>(pBf[boundary_label(patchName)])){
            vectorField normals = mesh.Sf().boundaryField()[boundary_label(patchName)]/mesh.magSf().boundaryField()[boundary_label(patchName)];

            refCast<timeVaryingPressureGradientFvPatchScalarField>(pBf[boundary_label(patchName)]).updateCoeffs
            (
                  gradP & normals
            );

      }
}


// ==================================================
// Boundary Specific Methods
// ==================================================
label cppFoamSolver::boundary_label(char *patchName){
      checkPatch(patchName);
      dynamicFvMesh& mesh= *meshPtr;
      return mesh.boundaryMesh().findPatchID(patchName);
}