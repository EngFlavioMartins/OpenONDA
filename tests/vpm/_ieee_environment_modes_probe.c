/* Qualification-only public-fenv probe. No FE values or fenv_t layout cross
 * the Python boundary; no production API or runtime compilation dependency.
 */
#include <fenv.h>
#include <stdlib.h>

#pragma STDC FENV_ACCESS ON

static int selected_rounding(int mode, int *value) {
    switch (mode) {
#ifdef FE_TONEAREST
        case 0: *value = FE_TONEAREST; return 0;
#endif
#ifdef FE_DOWNWARD
        case 1: *value = FE_DOWNWARD; return 0;
#endif
#ifdef FE_UPWARD
        case 2: *value = FE_UPWARD; return 0;
#endif
#ifdef FE_TOWARDZERO
        case 3: *value = FE_TOWARDZERO; return 0;
#endif
        default: return 1;
    }
}

int openonda_modes_enter(int mode, void **handle) {
    fenv_t *saved;
    fenv_t held;
    int rounding;
    if (!handle) return 1;
    *handle = NULL;
    if (selected_rounding(mode, &rounding)) return 2;
    saved = malloc(sizeof(*saved));
    if (!saved) return 3;
    if (fegetenv(saved)) { free(saved); return 4; }
    /* Standard non-stop mode avoids enabling traps when raising test flags. */
    if (feholdexcept(&held) || fesetround(rounding) || feclearexcept(FE_ALL_EXCEPT)
            || feraiseexcept(FE_INVALID | FE_DIVBYZERO)) {
        int restored = fesetenv(saved);
        free(saved);
        return restored ? 6 : 5;
    }
    *handle = saved;
    return 0;
}

int openonda_modes_observe(int *mode, int *flags) {
    int current, value, candidate;
    if (!mode || !flags) return 1;
    current = fegetround();
    *mode = -1;
    for (candidate = 0; candidate < 4; ++candidate)
        if (!selected_rounding(candidate, &value) && value == current)
            *mode = candidate;
    value = fetestexcept(FE_ALL_EXCEPT);
    *flags = ((value & FE_INVALID) ? 1 : 0)
           | ((value & FE_DIVBYZERO) ? 2 : 0)
           | ((value & FE_OVERFLOW) ? 4 : 0)
           | ((value & FE_UNDERFLOW) ? 8 : 0)
           | ((value & FE_INEXACT) ? 16 : 0);
    return *mode < 0 ? 2 : 0;
}

int openonda_modes_raise_underflow(void) {
    return feraiseexcept(FE_UNDERFLOW | FE_INEXACT);
}

int openonda_modes_restore(void *handle) {
    int status;
    if (!handle) return 1;
    status = fesetenv((const fenv_t *)handle);
    if (!status) free(handle);
    return status;
}
