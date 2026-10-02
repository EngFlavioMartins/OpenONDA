/* Qualification-only public fenv bridge; no solver runtime dependency. */
#include <fenv.h>
#include <stdlib.h>

#pragma STDC FENV_ACCESS ON

int openonda_qualification_fp_enter(void **handle) {
    fenv_t *saved;
    if (!handle) return 1;
    *handle = NULL;
    saved = malloc(sizeof(*saved));
    if (!saved) return 2;
    if (fegetenv(saved)) { free(saved); return 3; }
    if (fesetenv(FE_DFL_ENV)) {
        int restored = fesetenv(saved);
        free(saved);
        return restored ? 5 : 4;
    }
    *handle = saved;
    return 0;
}

int openonda_qualification_fp_restore(void *handle) {
    int status;
    if (!handle) return 1;
    status = fesetenv((const fenv_t *)handle);
    free(handle);
    return status;
}
