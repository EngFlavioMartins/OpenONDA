#ifndef FOAM_SOLVER_CORE_H
#define FOAM_SOLVER_CORE_H

#include "Time.H"
#include "fvCFD.H"
#include "dynamicFvMesh.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "pimpleControl.H"
#include "CorrectPhi.H"
#include "fvOptions.H"
#include "localEulerDdtScheme.H"
#include "fvcSmooth.H"
#include "autoPtr.H" 

#include "fluxControlledPressureFvPatchScalarField.H"
#include "timeVaryingPressureGradientFvPatchScalarField.H"
#include "timeVaryingInletOutletFvPatchField.H"
#include "timeVaryingOutletInletFvPatchField.H"

/*---------------------------------------------------------------------------*\
CLASS DEFINITION
\*---------------------------------------------------------------------------*/

class cppFoamSolver {
private:
    // Attributes
    int argc;
    char** argv;
    Foam::argList* _args;
    Foam::Time* _runTime;
    autoPtr<dynamicFvMesh> _meshPtr;
    dynamicFvMesh* meshPtr;

    autoPtr<dynamicFvMesh> _tempmeshPtr;
    dynamicFvMesh* tempmeshPtr;

    scalar cumulativeContErr;

    pimpleControl* _pimple;

    bool adjustTimeStep;
    scalar maxCo;
    scalar maxDeltaT;

    bool* _correctPhi;
    bool* _checkMeshCourantNo;
    bool* _moveMeshOuterCorrectors;

    bool LTS;
    tmp<volScalarField> trDeltaT;

    volScalarField* _p;
    volVectorField* _U;
    IOdictionary* _transportProperties;
    dimensionedScalar* _nu;
    surfaceScalarField* _phi;

    singlePhaseTransportModel* _laminarTransport;
    autoPtr<incompressible::turbulenceModel>* _turbulence;
    IOMRFZoneList* _MRF;

    autoPtr<surfaceVectorField> Uf;

    scalar* CoNumPtr;
    scalar* meanCoNumPtr;
    scalarField* sumPhiPtr;

    volVectorField* _vorticity;

    label numBoundLabel;

    vectorField* _inletRefValue;
    vectorField* _inletRefGradient;
    vectorField* _outletRefGradient;
    scalarField* _outletRefValue;

public:
    // ==================================================
    // Constructors and Destructor
    // ==================================================
    cppFoamSolver(int, char**);
    ~cppFoamSolver();

    // ==================================================
    // Simulation Methods
    // ==================================================
    void checkPatch(char *);
    void evolve();
    void evolve_mesh();
    void evolve_only_solution();
    void correct_mass_flux(char *);

    // ==================================================
    // Get Methods (Field and Mesh Data)
    // ==================================================
    double get_run_time_value();
    double get_time_step();
    
    int get_number_of_nodes();
    int get_number_of_cells();
    int get_number_of_boundary_nodes(char *);
    int get_number_of_boundary_faces(char *);

    // Cell and Node Coordinate Access
    void get_node_coordinates(double *);
    void get_connectivity(int *);
    void get_cell_volumes(double *);
    void get_cell_center_coordinates(double *);
    
    // Boundary Data Access
    void get_boundary_node_coordinates(double *, char *);
    void get_boundary_node_normal(double *, char *);
    void get_boundary_face_center_coordinates(double *, char *);
    void get_boundary_face_areas(double *, char *);
    void get_boundary_face_normals(double *, char *);
    void get_boundary_cell_center_coordinates(double *, char *);
    
    // Field Access Methods
    void get_velocity_field(double *);
    void get_velocity_boundary_field(double *, char *);
    void get_pressure_field(double *);
    void get_velocity_gradient(double *);
    void get_velocity_gradient_boundary_field(double *, char *);
    void get_pressure_gradient_field(double *);
    void get_pressure_boundary_field(double *, char *);
    void get_pressure_gradient_boundary_field(double *, char *);
    void get_vorticity_field(double *);
    void get_vorticity_boundary_field(double *, char *);

    // ==================================================
    // Set Methods (Boundary and Simulation Data)
    // ==================================================
    void set_time_step(double *);
    void set_dirichlet_velocity_boundary_condition(double *, char *);
    void set_dirichlet_pressure_boundary_condition(double *, char *);
    void set_neumann_pressure_boundary_condition(double *, char *);

    // ==================================================
    // Boundary Specific Methods
    // ==================================================
    label boundary_label(char *);
};

#endif
