cdef extern from "foamSolverCore.H":
      
      cdef cppclass cppFoamSolver:
            
            cppFoamSolver(int, char**)
            
            # ================================================== #
            # Simulation methods:
            # ================================================== #
            void evolve()
            void evolve_mesh()
            void evolve_only_solution()
            void correct_mass_flux(char *)

            # ==================================================
            # Simulation Methods
            # ==================================================
            double get_run_time_value()
            double get_time_step()
      
            int get_number_of_nodes()
            int get_number_of_cells()
            int get_number_of_boundary_nodes(char *)
            int get_number_of_boundary_faces(char *)

            # Cell and Node Coordinate Access
            void get_node_coordinates(double *)
            void get_connectivity(int *)
            void get_cell_volumes(double *)
            void get_cell_center_coordinates(double *)
            
            # Boundary Data Access
            void get_boundary_node_coordinates(double *, char *)
            void get_boundary_node_normal(double *, char *)
            void get_boundary_face_center_coordinates(double *, char *)
            void get_boundary_face_areas(double *, char *)
            void get_boundary_face_normals(double *, char *)
            void get_boundary_cell_center_coordinates(double *, char *)
            
            # Field Access Methods
            void get_velocity_field(double *)
            void get_velocity_boundary_field(double *, char *)
            void get_pressure_field(double *)
            void get_velocity_gradient(double *)
            void get_velocity_gradient_boundary_field(double *, char *)
            void get_pressure_gradient_field(double *)
            void get_pressure_boundary_field(double *, char *)
            void get_pressure_gradient_boundary_field(double *, char *)
            void get_vorticity_field(double *)
            void get_vorticity_boundary_field(double *, char *)

            # ================================================== #
            # Set Methods (Boundary and Simulation Data)
            # ================================================== #
            void set_time_step(double *)
            void set_dirichlet_velocity_boundary_condition(double *, char *)
            void set_dirichlet_pressure_boundary_condition(double *, char *)
            void set_neumann_pressure_boundary_condition(double *, char *)

            #label boundary_label(char *)
            
      
            # =====================================
            # Python-based functions:
            # =====================================
            # faceVelocityX2, faceVelocityY2, faceVelocityZ2 = correct_mass_flux_python(double *, faceVelocityX, faceVelocityY, faceVelocityZ, patchName="numericalBoundary")
            
            # dpdx, dpdy, dpdz = correct_normal_pressure_gradient(self, dpdx, dpdy, dpdz, patchName="numericalBoundary")
            
            # np.array([centroid_x,centroid_y,centroid_z])  = get_mesh_centroid()