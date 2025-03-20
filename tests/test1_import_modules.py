import pytest

# Run this using: pytest test.py

# Test importing the main package
def test_import_OpenONDA():
    try:
        import OpenONDA
    except Exception as e:
        pytest.fail(f"Failed to import OpenONDA: {e}")

# Test importing submodules (FVM, VPM, and utilities)
def test_import_submodules():
    """Test if all major modules can be imported successfully."""
    try:
        from OpenONDA.solvers.FVM import fvmModule  # Import fvmModule from FVM
        from OpenONDA.solvers.VPM import vpmModule  # Import vpmModule from VPM
        from OpenONDA.utilities import set_initial_condition  # Import utility functions
    except Exception as e:
        pytest.fail(f"Module import error: {e}")

# Test ParticleSystem functionality
def test_particle_system():
    """Test the basic functionality of ParticleSystem."""
    try:
        from OpenONDA.solvers.VPM import vpmModule  # Import vpmModule from VPM
        
        # Define required parameters
        flow_model = 'LES'
        time_step_size = 0.01  # example value, adjust as needed
        time_integration_method = 'RK2'
        viscous_scheme = 'CoreSpreading'
        processing_unit = 'GPU'
        monitor_variables = ['Circulation', 'Kinetic energy']
        backup_filename = 'particle_system_backup.dat'
        backup_frequency = 10
        
        # Create ParticleSystem instance with necessary arguments
        ps = vpmModule.ParticleSystem(
            flow_model=flow_model,
            time_step_size=time_step_size,
            time_integration_method=time_integration_method,
            viscous_scheme=viscous_scheme,
            processing_unit=processing_unit,
            monitor_variables=monitor_variables,
            backup_filename=backup_filename,
            backup_frequency=backup_frequency
        )
        
        # Check that the instance was created successfully
        assert ps is not None, "ParticleSystem instance is None"
        
        # Optionally, check some attributes to ensure the object is initialized correctly
        assert ps.flow_model == flow_model, f"Expected {flow_model}, got {ps.flow_model}"
        assert ps.time_step_size == time_step_size, f"Expected {time_step_size}, got {ps.time_step_size}"
        
    except Exception as e:
        pytest.fail(f"ParticleSystem test failed: {e}")


if __name__ == "__main__":
    pytest.main()
