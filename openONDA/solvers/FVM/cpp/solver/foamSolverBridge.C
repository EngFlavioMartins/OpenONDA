#include "foamSolverCore.H"
#include "foamSolverBridge.H"

extern "C"
{
    // ==================================================
    // Constructors and Destructor
    // ==================================================
    pimpleStepperFoam* pimpleStepperFoam_create(int argc, char *argv[]){
        return reinterpret_cast<pimpleStepperFoam*>(new cppFoamSolver(argc, argv));
    }

    void pimpleStepperFoam_delete(pimpleStepperFoam* solver){
        delete reinterpret_cast<cppFoamSolver*>(solver);
    }


    // ==================================================
    // Simulation Methods
    // ==================================================
    void pimpleStepperFoam_checkPatch(pimpleStepperFoam* solver, char* patchName)
    {
        reinterpret_cast<cppFoamSolver*>(solver)->checkPatch(patchName);
    }


    void pimpleStepperFoam_evolve(pimpleStepperFoam* solver){
        reinterpret_cast<cppFoamSolver*>(solver)->evolve();
    }

    void pimpleStepperFoam_evolve_mesh(pimpleStepperFoam* solver){
        reinterpret_cast<cppFoamSolver*>(solver)->evolve_mesh();
    }

    void pimpleStepperFoam_evolve_only_solution(pimpleStepperFoam* solver){
        reinterpret_cast<cppFoamSolver*>(solver)->evolve_only_solution();
    }

    void pimpleStepperFoam_correct_mass_flux(pimpleStepperFoam* solver, char *patchName){
        reinterpret_cast<cppFoamSolver*>(solver)->correct_mass_flux(patchName);
    }


    // ==================================================
    // Get Methods (Field and Mesh Data)
    // ==================================================
    double pimpleStepperFoam_get_run_time_value(pimpleStepperFoam* solver){
        return reinterpret_cast<cppFoamSolver*>(solver)->get_run_time_value();
    }

    double pimpleStepperFoam_get_time_step(pimpleStepperFoam* solver){
        return reinterpret_cast<cppFoamSolver*>(solver)->get_time_step();
    }

    int pimpleStepperFoam_get_number_of_nodes(pimpleStepperFoam* solver){
        return reinterpret_cast<cppFoamSolver*>(solver)->get_number_of_nodes();
    }

    int pimpleStepperFoam_get_number_of_cells(pimpleStepperFoam* solver){
        return reinterpret_cast<cppFoamSolver*>(solver)->get_number_of_cells();
    }

    int pimpleStepperFoam_get_number_of_boundary_nodes(pimpleStepperFoam* solver, char *patchName){
        return reinterpret_cast<cppFoamSolver*>(solver)->get_number_of_boundary_nodes(patchName);
    }

    int pimpleStepperFoam_get_number_of_boundary_faces(pimpleStepperFoam* solver,char *patchName){
        return reinterpret_cast<cppFoamSolver*>(solver)->get_number_of_boundary_faces(patchName);
    }


    // Cell and Node Coordinate Access
    void pimpleStepperFoam_get_node_coordinates(pimpleStepperFoam* solver, double *coordinates){
        reinterpret_cast<cppFoamSolver*>(solver)->get_node_coordinates(coordinates);
    }

    void pimpleStepperFoam_get_connectivity(pimpleStepperFoam* solver, int *connectivity){
        reinterpret_cast<cppFoamSolver*>(solver)->get_connectivity(connectivity);
    }

    
    void pimpleStepperFoam_get_cell_volume(pimpleStepperFoam* solver, double *volume){
        return reinterpret_cast<cppFoamSolver*>(solver)->get_cell_volumes(volume);
    }

    void pimpleStepperFoam_get_cell_center_coordinates(pimpleStepperFoam* solver, double *coordinates){
        reinterpret_cast<cppFoamSolver*>(solver)->get_cell_center_coordinates(coordinates);
    }


    // Boundary Data Access
    void pimpleStepperFoam_get_boundary_node_coordinates(pimpleStepperFoam* solver, double *coordinates, char *patchName){
        reinterpret_cast<cppFoamSolver*>(solver)->get_boundary_node_coordinates(coordinates, patchName);
    }

    void pimpleStepperFoam_get_boundary_node_normal(pimpleStepperFoam* solver, double *normals, char *patchName){
        reinterpret_cast<cppFoamSolver*>(solver)->get_boundary_node_normal(normals, patchName);
    }

    void pimpleStepperFoam_get_boundary_face_center_coordinates(pimpleStepperFoam* solver, double *coordinates, char *patchName){
        reinterpret_cast<cppFoamSolver*>(solver)->get_boundary_face_center_coordinates(coordinates, patchName);
    }

    void pimpleStepperFoam_get_boundary_face_areas(pimpleStepperFoam* solver, double *areas,char *patchName){
        reinterpret_cast<cppFoamSolver*>(solver)->get_boundary_face_areas(areas, patchName);
    }

    void pimpleStepperFoam_get_boundary_face_normals(pimpleStepperFoam* solver, double *normals,char *patchName){
        reinterpret_cast<cppFoamSolver*>(solver)->get_boundary_face_normals(normals, patchName);
    }

    void pimpleStepperFoam_get_boundary_cell_center_coordinates(pimpleStepperFoam* solver,double *coordinates,char *patchName){
        reinterpret_cast<cppFoamSolver*>(solver)->get_boundary_cell_center_coordinates(coordinates, patchName);
    }


    // Field Access Methods
    void pimpleStepperFoam_get_velocity_field(pimpleStepperFoam* solver, double *pyVelocity){
        reinterpret_cast<cppFoamSolver*>(solver)->get_velocity_field(pyVelocity);
    }

    void pimpleStepperFoam_get_velocity_boundary_field(pimpleStepperFoam* solver, double *pyVelocity,char *patchName){
        reinterpret_cast<cppFoamSolver*>(solver)->get_velocity_boundary_field(pyVelocity, patchName);
    }

    void pimpleStepperFoam_get_pressure_field(pimpleStepperFoam* solver, double *pyPressure){
        reinterpret_cast<cppFoamSolver*>(solver)->get_pressure_field(pyPressure);
    }

    void pimpleStepperFoam_get_velocity_gradient(pimpleStepperFoam* solver, double *pyVelocityGradient){
        reinterpret_cast<cppFoamSolver*>(solver)->get_velocity_gradient(pyVelocityGradient);
    }

    // TODO: Test
    void get_velocity_gradient_boundary_field(pimpleStepperFoam* solver, double *pyVelocityGradient,  char *patchName){
        reinterpret_cast<cppFoamSolver*>(solver)->get_velocity_gradient_boundary_field(pyVelocityGradient, patchName);
    }
    
    void pimpleStepperFoam_get_pressure_gradient_field(pimpleStepperFoam* solver, double *pyPressureGradient){
        reinterpret_cast<cppFoamSolver*>(solver)->get_pressure_gradient_field(pyPressureGradient);
    }

    void pimpleStepperFoam_get_pressure_boundary_field(pimpleStepperFoam* solver, double *pyPressure, char *patchName){
        reinterpret_cast<cppFoamSolver*>(solver)->get_pressure_boundary_field(pyPressure, patchName);
    }

    void pimpleStepperFoam_get_pressure_gradient_boundary_field(pimpleStepperFoam* solver,double *pyPressureGradient, char *patchName){
        reinterpret_cast<cppFoamSolver*>(solver)->get_pressure_gradient_boundary_field(pyPressureGradient, patchName);
    }

    void pimpleStepperFoam_get_vorticity_field(pimpleStepperFoam* solver, double *pyVorticity){
        reinterpret_cast<cppFoamSolver*>(solver)->get_vorticity_field(pyVorticity);
    }

    void pimpleStepperFoam_get_vorticity_boundary_field(pimpleStepperFoam* solver, double *pyVorticity, char *patchName){
        reinterpret_cast<cppFoamSolver*>(solver)->get_vorticity_boundary_field(pyVorticity, patchName);
    }


    // ==================================================
    // Set Methods (Boundary and Simulation Data)
    // ==================================================
    void pimpleStepperFoam_set_time_step(pimpleStepperFoam* solver, double *tStep){
        reinterpret_cast<cppFoamSolver*>(solver)->set_time_step(tStep);
    }
    
    void pimpleStepperFoam_set_dirichlet_velocity_boundary_condition(pimpleStepperFoam* solver, double *pyVelocityBC, char *patchName){
        reinterpret_cast<cppFoamSolver*>(solver)->set_dirichlet_velocity_boundary_condition(pyVelocityBC, patchName);
    }

    void pimpleStepperFoam_set_dirichlet_pressure_boundary_condition(pimpleStepperFoam* solver, double *pyPressureBC, char *patchName){
        reinterpret_cast<cppFoamSolver*>(solver)->set_dirichlet_pressure_boundary_condition(pyPressureBC, patchName);
    }

    void pimpleStepperFoam_set_neumann_pressure_boundary_condition(pimpleStepperFoam* solver, double *pyPressureGradientBC, char *patchName){
        reinterpret_cast<cppFoamSolver*>(solver)->set_neumann_pressure_boundary_condition(pyPressureGradientBC, patchName);
    }

}