#include "fmm_c.h"

#if FMM_ERRORCONTROL_STAGE > 0
#  if FMM_MAXNMULTIPOLES > 39
   const fmm_float_t fmmec_3d_p40[] = {
#  include "include/fmmec_3d_p40.h"
   };

   const fmm_float_t *get_fmmec_3d_p40()
   {
     return fmmec_3d_p40;
   }
#  endif
#endif
