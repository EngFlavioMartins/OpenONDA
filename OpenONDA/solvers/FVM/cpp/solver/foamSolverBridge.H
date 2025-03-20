#ifndef __FOAM_SOLVER_BRIDGER__
#define __FOAM_SOLVER_BRIDGER__

#ifdef __cplusplus

extern "C" 
{
      #endif //__cplusplus

      struct HpimpleStepperFoam;

      typedef struct HpimpleStepperFoam pimpleStepperFoam;

      // ==================================================
      // Constructors and Destructor
      // ==================================================
      pimpleStepperFoam* pimpleStepperFoam_create(int argc, char *argv[]);
      void pimpleStepperFoam_delete(pimpleStepperFoam*);

      // ==================================================
      // Simulation Methods
      // ==================================================
      void pimpleStepperFoam_checkPatch(pimpleStepperFoam*, char*);
      void pimpleStepperFoam_evolve(pimpleStepperFoam*);
      void pimpleStepperFoam_evolve_mesh(pimpleStepperFoam*);
      void pimpleStepperFoam_evolve_only_solution(pimpleStepperFoam*);
      void pimpleStepperFoam_correct_mass_flux(pimpleStepperFoam*, char *);

      // ==================================================
      // "Get" methods:
      // ==================================================
      double pimpleStepperFoam_get_run_time_value(pimpleStepperFoam*);
      double pimpleStepperFoam_get_time_step(pimpleStepperFoam*);
      
      int pimpleStepperFoam_get_number_of_nodes(pimpleStepperFoam*);
      int pimpleStepperFoam_get_number_of_cells(pimpleStepperFoam*);
      int pimpleStepperFoam_get_number_of_boundary_nodes(pimpleStepperFoam*, char *);
      int pimpleStepperFoam_get_number_of_boundary_faces(pimpleStepperFoam*, char *);
      
      // Cell and Node Coordinate Access
      void pimpleStepperFoam_get_node_coordinates(pimpleStepperFoam*, double *);
      void pimpleStepperFoam_get_connectivity(pimpleStepperFoam*, int *);
      void pimpleStepperFoam_get_cell_volume(pimpleStepperFoam*, double *);
      void pimpleStepperFoam_get_cell_center_coordinates(pimpleStepperFoam*, double *);
      
      // Boundary Data Access
      void pimpleStepperFoam_get_boundary_node_coordinates(pimpleStepperFoam*, double *, char *);
      void pimpleStepperFoam_get_boundary_node_normal(pimpleStepperFoam*, double *, char *);
      void pimpleStepperFoam_get_boundary_face_center_coordinates(pimpleStepperFoam*, double *, char *);
      void pimpleStepperFoam_get_boundary_face_areas(pimpleStepperFoam*, double *, char *);
      void pimpleStepperFoam_get_boundary_face_normals(pimpleStepperFoam*, double *, char *);
      void pimpleStepperFoam_get_boundary_cell_center_coordinates(pimpleStepperFoam*, double *, char *);

      // Field Access Methods
      void pimpleStepperFoam_get_velocity_field(pimpleStepperFoam*, double *);
      void pimpleStepperFoam_get_velocity_boundary_field(pimpleStepperFoam*, double *, char *);
      void pimpleStepperFoam_get_pressure_field(pimpleStepperFoam*, double *);
      void pimpleStepperFoam_get_velocity_gradient(pimpleStepperFoam*, double *);
      void pimpleStepperFoam_get_velocity_gradient_boundary_field(pimpleStepperFoam*, double *, char *);
      void pimpleStepperFoam_get_pressure_boundary_field(pimpleStepperFoam*, double *, char *);
      void pimpleStepperFoam_get_pressure_gradient_field(pimpleStepperFoam*, double *);
      void pimpleStepperFoam_get_pressure_gradient_boundary_field(pimpleStepperFoam*, double *, char *);
      void pimpleStepperFoam_get_vorticity_field(pimpleStepperFoam*, double *);
      void pimpleStepperFoam_get_vorticity_boundary_field(pimpleStepperFoam*, double *, char *);

      // ==================================================
      // Set Methods (Boundary and Simulation Data)
      // ==================================================
      void pimpleStepperFoam_set_time_step(pimpleStepperFoam*, double *); 

      void pimpleStepperFoam_set_dirichlet_velocity_boundary_condition(pimpleStepperFoam*, double *, char *);
      void pimpleStepperFoam_set_dirichlet_pressure_boundary_condition(pimpleStepperFoam*, double *, char *);
      void pimpleStepperFoam_set_neumann_pressure_boundary_condition(pimpleStepperFoam*, double *, char *);

      #ifdef __cplusplus
}

#endif //__cplusplus

#endif //__VORTICITY_FOAM_WRAPPER__
