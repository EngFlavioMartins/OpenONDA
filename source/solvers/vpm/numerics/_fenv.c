/* Optional precompiled bridge to the public C floating-point environment.
 * Do not assume a platform fenv_t layout or encode FE_DFL_ENV in Python.
 * Native bookkeeping makes mode/stack transitions indivisible to Python
 * exceptions. No numerical work occurs here.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <fenv.h>

#pragma STDC FENV_ACCESS ON

typedef struct {
    fenv_t environment;
    uint64_t thread_state;
    int state; /* 0: captured token only, 1: active, 2: restored */
    PyObject *previous;
} SavedEnvironment;

static const char *token_name = "openonda.saved_fenv";
static Py_tss_t active_scope = Py_tss_NEEDS_INIT;

static void token_destroy(PyObject *capsule) {
    SavedEnvironment *saved = PyCapsule_GetPointer(capsule, token_name);
    /* An active capsule has an owned native TLS reference, so asynchronous
     * Python finalization cannot destroy its saved environment. */
    if (saved) PyMem_Free(saved);
    else PyErr_Clear();
}

static PyObject *round_to_nearest(PyObject *self, PyObject *unused) {
    int mode = fegetround();
    if (mode < 0)
        return PyErr_Format(PyExc_RuntimeError, "Cannot inspect floating-point rounding mode");
    return PyBool_FromLong(mode == FE_TONEAREST);
}

static PyObject *capture(PyObject *self, PyObject *unused) {
    SavedEnvironment *saved = PyMem_Calloc(1, sizeof(*saved));
    PyObject *capsule;
    if (!saved) return PyErr_NoMemory();
    capsule = PyCapsule_New(saved, token_name, token_destroy);
    if (!capsule) { PyMem_Free(saved); return NULL; }
    saved->thread_state = PyThreadState_GetID(PyThreadState_Get());
    return capsule; /* No floating-point state has changed. */
}

static PyObject *enter_default(PyObject *self, PyObject *capsule) {
    SavedEnvironment *saved = PyCapsule_GetPointer(capsule, token_name);
    if (!saved) return NULL;
    if (saved->state)
        return PyErr_Format(PyExc_RuntimeError, "Floating-point token is single use");
    if (saved->thread_state != PyThreadState_GetID(PyThreadState_Get()))
        return PyErr_Format(PyExc_RuntimeError, "Floating-point environment belongs to another thread");
    if (fegetenv(&saved->environment)) {
        return PyErr_Format(PyExc_RuntimeError, "Cannot capture floating-point environment");
    }
    saved->previous = PyThread_tss_get(&active_scope);
    if (PyThread_tss_set(&active_scope, capsule)) {
        saved->previous = NULL;
        return PyErr_Format(PyExc_RuntimeError, "Cannot retain floating-point scope");
    }
    Py_INCREF(capsule); /* TLS references current; the previous TLS reference moves to saved. */
    saved->state = 1;
    if (fesetenv(FE_DFL_ENV)) {
        /* The caller's already-established finally will restore this active
         * token even if entry failed after partially changing the mode. */
        return PyErr_Format(PyExc_RuntimeError, "Cannot enter default floating-point environment");
    }
    Py_RETURN_NONE;
}

static PyObject *restore(PyObject *self, PyObject *capsule) {
    SavedEnvironment *saved = PyCapsule_GetPointer(capsule, token_name);
    if (!saved) return NULL;
    if (saved->thread_state != PyThreadState_GetID(PyThreadState_Get()))
        return PyErr_Format(PyExc_RuntimeError, "Floating-point environment belongs to another thread");
    /* Idempotent cleanup also handles an exception before successful entry. */
    if (saved->state != 1) Py_RETURN_NONE;
    if (PyThread_tss_get(&active_scope) != capsule)
        return PyErr_Format(PyExc_RuntimeError, "Floating-point scopes must restore in reverse entry order");
    if (PyThread_tss_set(&active_scope, saved->previous))
        return PyErr_Format(PyExc_RuntimeError, "Cannot release floating-point scope");
    if (fesetenv(&saved->environment)) {
        if (PyThread_tss_set(&active_scope, capsule))
            return PyErr_Format(PyExc_RuntimeError, "Cannot restore floating-point environment or thread context");
        return PyErr_Format(PyExc_RuntimeError, "Cannot restore floating-point environment");
    }
    saved->previous = NULL; /* Previous owned reference moves back to TLS. */
    saved->state = 2;
    Py_DECREF(capsule);
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"round_to_nearest", round_to_nearest, METH_NOARGS,
     "Inspect FE_TONEAREST without changing rounding, exception flags or underflow modes."},
    {"capture", capture, METH_NOARGS, "Allocate an inactive same-thread token without changing FENV."},
    {"enter_default", enter_default, METH_O, "Retain a token and enter FE_DFL_ENV atomically."},
    {"restore", restore, METH_O, "Restore a same-thread LIFO token; inactive cleanup is a no-op."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_fenv", "Optional public-fenv bridge.", -1, methods
};

PyMODINIT_FUNC PyInit__fenv(void) {
    if (PyThread_tss_create(&active_scope))
        return PyErr_Format(PyExc_RuntimeError, "Cannot create floating-point scope storage");
    return PyModule_Create(&module);
}
