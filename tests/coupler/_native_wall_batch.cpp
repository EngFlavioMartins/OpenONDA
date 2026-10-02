/* UNWIRED qualification prototype: exact VTK point queries, private workers.
 * Compile against the SAME VTK headers/libraries as the Python installation.
 * No SMP setting, process affinity, Python object address or global allocator
 * is changed. The caller's existing domain clipping remains outside this API.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <vtkCellType.h>
#include <vtkImplicitPolyDataDistance.h>
#include <vtkPolyData.h>
#include <vtkPythonUtil.h>
#include <vtkSmartPointer.h>
#include <vtkVersion.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
#ifdef __linux__
#include <sched.h>
#endif

namespace {
constexpr const char* capsule_name = "openonda.test_native_wall_batch";
struct Owner {
  uint64_t thread_state = PyThreadState_GetID(PyThreadState_Get());
  bool closed = false;
  Py_ssize_t max_points = 0;
  int workers = 0;
  vtkIdType triangles = 0;
  vtkSmartPointer<vtkPolyData> surface;
  std::vector<vtkSmartPointer<vtkImplicitPolyDataDistance>> functions;
};

int affinity_count() {
#ifdef __linux__
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) == 0)
    return std::max(1, CPU_COUNT(&mask));
#endif
  // No proof of a larger process allocation: keep the conservative serial path.
  return 1;
}

void destroy(PyObject* capsule) {
  auto* owner = static_cast<Owner*>(PyCapsule_GetPointer(capsule, capsule_name));
  if (owner) delete owner; else PyErr_Clear();
}

Owner* get_owner(PyObject* capsule, bool admit_closed = false) {
  auto* owner = static_cast<Owner*>(PyCapsule_GetPointer(capsule, capsule_name));
  if (!owner) return nullptr;
  if (owner->thread_state != PyThreadState_GetID(PyThreadState_Get())) {
    PyErr_SetString(PyExc_RuntimeError, "wall query owner belongs to another Python thread");
    return nullptr;
  }
  if (owner->closed && !admit_closed) {
    PyErr_SetString(PyExc_RuntimeError, "wall query owner is closed");
    return nullptr;
  }
  return owner;
}

PyObject* translate_exception() {
  try { throw; }
  catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
  catch (const std::exception& error) { PyErr_SetString(PyExc_RuntimeError, error.what()); }
  catch (...) { PyErr_SetString(PyExc_RuntimeError, "unknown native wall-query failure"); }
  return nullptr;
}

PyObject* create(PyObject*, PyObject* args) {
  PyObject *surface_object, *distance_object;
  int requested;
  Py_ssize_t max_points, max_triangles;
  if (!PyArg_ParseTuple(args, "OOinn", &surface_object, &distance_object,
                        &requested, &max_points, &max_triangles)) return nullptr;
  if (PyBool_Check(PyTuple_GetItem(args, 2)) || PyBool_Check(PyTuple_GetItem(args, 3)) ||
      PyBool_Check(PyTuple_GetItem(args, 4)) ||
      requested < 1 || requested > 64 || max_points < 1 || max_points > 1000000 ||
      max_triangles < 1 || max_triangles > 1000000) {
    PyErr_SetString(PyExc_ValueError, "bounded worker/point/triangle limits required");
    return nullptr;
  }
  auto* surface = vtkPolyData::SafeDownCast(
      vtkPythonUtil::GetPointerFromObject(surface_object, "vtkPolyData"));
  if (!surface) return nullptr;
  auto* distance = vtkImplicitPolyDataDistance::SafeDownCast(
      vtkPythonUtil::GetPointerFromObject(distance_object, "vtkImplicitPolyDataDistance"));
  if (!distance) return nullptr;
  if (std::strcmp(surface->GetClassName(), "vtkPolyData") ||
      std::strcmp(distance->GetClassName(), "vtkImplicitPolyDataDistance") ||
      distance->GetTransform() != nullptr || surface->GetNumberOfPolys() < 1 ||
      surface->GetNumberOfPolys() > max_triangles ||
      surface->GetNumberOfCells() != surface->GetNumberOfPolys() ||
      surface->GetNumberOfPoints() < 3 ||
      surface->GetNumberOfPoints() > 3*max_triangles) {
    PyErr_SetString(PyExc_ValueError, "standard finite polygon surface without transform required");
    return nullptr;
  }
  try {
    auto owner = std::make_unique<Owner>();
    owner->max_points = max_points;
    owner->workers = std::min(requested, affinity_count());
    owner->triangles = surface->GetNumberOfPolys();
    owner->surface = vtkSmartPointer<vtkPolyData>::New();
    owner->surface->DeepCopy(surface);
    for (vtkIdType cell = 0; cell < owner->surface->GetNumberOfCells(); ++cell)
      if (owner->surface->GetCellType(cell) != VTK_TRIANGLE)
        throw std::invalid_argument("only already-triangulated surfaces are admitted");
    for (vtkIdType p = 0; p < owner->surface->GetNumberOfPoints(); ++p) {
      double point[3]; owner->surface->GetPoint(p, point);
      if (!std::all_of(point, point+3, [](double x){ return std::isfinite(x); }))
        throw std::invalid_argument("finite surface coordinates required");
    }
    owner->functions.reserve(owner->workers);
    for (int i = 0; i < owner->workers; ++i) {
      auto function = vtkSmartPointer<vtkImplicitPolyDataDistance>::New();
      function->SetInput(owner->surface);
      function->SetTolerance(distance->GetTolerance());
      function->SetNoValue(distance->GetNoValue());
      function->SetNoGradient(distance->GetNoGradient());
      function->SetNoClosestPoint(distance->GetNoClosestPoint());
      // Initialize each independent locator before concurrent read-only use.
      double point[3]; owner->surface->GetPoint(0, point);
      function->FunctionValue(point);
      owner->functions.push_back(function);
    }
    PyObject* capsule = PyCapsule_New(owner.get(), capsule_name, destroy);
    if (capsule) owner.release();
    return capsule;
  } catch (...) { return translate_exception(); }
}

struct JoinThreads {
  std::vector<std::thread> threads;
  ~JoinThreads() { for (auto& thread : threads) if (thread.joinable()) thread.join(); }
};

PyObject* evaluate(PyObject*, PyObject* args) {
  PyObject *capsule, *array;
  int injected_worker = -1; // Qualification-only failure injection.
  if (!PyArg_ParseTuple(args, "OO|i", &capsule, &array, &injected_worker)) return nullptr;
  Owner* owner = get_owner(capsule);
  if (!owner) return nullptr;
  Py_buffer view{};
  if (PyObject_GetBuffer(array, &view, PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) < 0) return nullptr;
  const bool valid = view.ndim == 2 && view.shape[1] == 3 && view.shape[0] <= owner->max_points &&
      view.itemsize == sizeof(double) && view.format && !std::strcmp(view.format, "d") &&
      PyBuffer_IsContiguous(&view, 'C');
  if (!valid) {
    PyBuffer_Release(&view);
    PyErr_SetString(PyExc_ValueError, "bounded contiguous native float64 points (N,3) required");
    return nullptr;
  }
  const Py_ssize_t count = view.shape[0];
  std::vector<double> query, output;
  try {
    auto* begin = static_cast<const double*>(view.buf);
    if (count) query.assign(begin, begin+3*count);
    output.resize(count);
  } catch (...) { PyBuffer_Release(&view); return translate_exception(); }
  PyBuffer_Release(&view); // Caller memory is not read after releasing the GIL.
  if (!std::all_of(query.begin(), query.end(), [](double x){ return std::isfinite(x); })) {
    PyErr_SetString(PyExc_ValueError, "finite query coordinates required"); return nullptr;
  }
  if (injected_worker < -1 || injected_worker >= owner->workers) {
    PyErr_SetString(PyExc_ValueError, "invalid qualification failure worker"); return nullptr;
  }
  std::exception_ptr failure;
  // The argument tuple retains the capsule until this function returns. No
  // Python API occurs below; same-thread admission rejects simultaneous calls.
  PyThreadState* saved = PyEval_SaveThread();
  try {
    const int workers = static_cast<int>(std::min<Py_ssize_t>(owner->workers, count));
    std::vector<std::exception_ptr> failures(workers);
    {
      JoinThreads tasks;
      tasks.threads.reserve(workers);
      for (int worker = 0; worker < workers; ++worker) {
        tasks.threads.emplace_back([&, worker] {
          try {
            if (worker == injected_worker) throw std::runtime_error("injected worker failure");
            const Py_ssize_t first = count*worker/workers, last = count*(worker+1)/workers;
            for (Py_ssize_t row = first; row < last; ++row) {
              double value = owner->functions[worker]->FunctionValue(query.data()+3*row);
              if (!std::isfinite(value)) throw std::runtime_error("nonfinite native wall distance");
              output[row] = value;
            }
          } catch (...) { failures[worker] = std::current_exception(); }
        });
      }
    } // ALL started workers join, also if thread creation throws.
    for (const auto& candidate : failures) if (candidate) { failure = candidate; break; }
  } catch (...) { failure = std::current_exception(); }
  PyEval_RestoreThread(saved);
  if (failure) { try { std::rethrow_exception(failure); } catch (...) { return translate_exception(); } }
  // Private output is exposed only after every worker succeeded and joined.
  return PyBytes_FromStringAndSize(reinterpret_cast<const char*>(output.data()), count*sizeof(double));
}

PyObject* close(PyObject*, PyObject* capsule) {
  Owner* owner = get_owner(capsule, true);
  if (!owner) return nullptr;
  owner->closed = true;
  owner->functions.clear();
  owner->surface = nullptr;
  Py_RETURN_NONE;
}

PyObject* metadata(PyObject*, PyObject* capsule) {
  Owner* owner = get_owner(capsule, true);
  if (!owner) return nullptr;
  return Py_BuildValue("{s:s,s:i,s:n,s:L,s:O}", "vtk_version", GetVTKVersion(),
      "workers", owner->workers, "max_points", owner->max_points,
      "triangles", static_cast<long long>(owner->triangles), "closed", owner->closed ? Py_True : Py_False);
}

PyMethodDef methods[] = {
  {"create", create, METH_VARARGS, "Create independent immutable-mesh query owners."},
  {"evaluate", evaluate, METH_VARARGS, "Return private ordered f64 bytes after all workers join."},
  {"close", close, METH_O, "Idempotent same-thread owner teardown."},
  {"metadata", metadata, METH_O, "Read qualification ownership metadata."},
  {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native_wall_batch", "Unwired VTK batch prototype.", -1, methods};
}

PyMODINIT_FUNC PyInit__native_wall_batch() {
  if (std::strcmp(GetVTKVersion(), VTK_VERSION)) {
    PyErr_SetString(PyExc_ImportError, "compiled/runtime VTK version mismatch"); return nullptr;
  }
  return PyModule_Create(&module);
}
