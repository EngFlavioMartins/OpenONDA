import pytest

# Run this using: pytest test.py

# Try importing the main package and its modules
def test_import_openONDA():
    try:
        import openONDA
    except Exception as e:
        pytest.fail(f"Failed to import openONDA: {e}")

def test_import_submodules():
    """Test if all major modules can be imported successfully."""
    try:
        from openONDA import fvmModule, ParticleSystem
        from openONDA.solvers.VPM import ParticlesHelpers
        from openONDA.utilities import eulerian_solver_helper
    except Exception as e:
        pytest.fail(f"Module import error: {e}")

def test_particle_system():
    """Test the basic functionality of ParticleSystem."""
    try:
        from openONDA.solvers.VPM import ParticleSystem
        ps = ParticleSystem()  # Assuming it has a default constructor
        assert ps is not None, "ParticleSystem instance is None"
    except Exception as e:
        pytest.fail(f"ParticleSystem test failed: {e}")

if __name__ == "__main__":
    pytest.main()
