/*
  Copyright (C) 2011, 2012, 2013 Olaf Lenz, Rene Halver, Michael Hofmann
  Copyright (C) 2017 Dirk Leichsenring
  Copyright (C) 2018 Michael Hofmann
  
  This file is part of ScaFaCoS.
  
  ScaFaCoS is free software: you can redistribute it and/or modify
  it under the terms of the GNU Lesser Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  
  ScaFaCoS is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Lesser Public License for more details.
  
  You should have received a copy of the GNU Lesser Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __NEAR_COMMON_H__
#define __NEAR_COMMON_H__

#ifdef __cplusplus
extern "C" {
#endif

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#define FCS_NEAR_ASYNC            1
#define FCS_NEAR_OCL_ASYNC        1
#define FCS_NEAR_OCL_WAIT_WRITE   0
#define FCS_NEAR_BOX_IS_LONG_LONG 1

#define COMPUTE                   1
#define COMPUTE_BOX               1
#define COMPUTE_REAL_NEIGHBOURS   1
#define COMPUTE_GHOST_NEIGHBOURS  1
#define COMPUTE_VERBOSE           0

#define POTENTIAL_CONST1  0
#define PRINT_PARTICLES   0
#define PRINT_BOX_STATS   0

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


#include <mpi.h>

#if FCS_NEAR_ASYNC
# include <pthread.h>
#endif

#include "common/fcs-common/FCSCommon.h"

#if HAVE_OPENCL
#include "common/fcs-opencl/fcs_ocl.h"
#endif

#include "common/gridsort/gridsort.h"

#include "sl_near_fp.h"
#include "sl_near_f_.h"
#include "sl_near__p.h"
#include "sl_near___.h"

#include "z_tools.h"
#include "near.h"

#if HAVE_OPENCL
#include "near_ocl.h"
#include "near_sort.h"
#endif

#ifdef FCS_NEAR_BOX_IS_LONG_LONG
  typedef long long box_t;
#else
# error Type for box_t not available
#endif


#if defined(FCS_ENABLE_DEBUG_NEAR)
# define DO_DEBUG
# define DEBUG_CMD(_cmd_)  Z_MOP(_cmd_)
#else
# define DEBUG_CMD(_cmd_)  Z_NOP()
#endif
#define DEBUG_PRINT_PREFIX  "NEAR_DEBUG: "

#if defined(FCS_ENABLE_INFO_NEAR)
# define DO_INFO
# define INFO_CMD(_cmd_)  Z_MOP(_cmd_)
#else
# define INFO_CMD(_cmd_)  Z_NOP()
#endif
#define INFO_PRINT_PREFIX  "NEAR_INFO: "

#if defined(FCS_ENABLE_TIMING_NEAR)
# define DO_TIMING
# define TIMING_CMD(_cmd_)  Z_MOP(_cmd_)
#else
# define TIMING_CMD(_cmd_)  Z_NOP()
#endif
#define TIMING_PRINT_PREFIX  "NEAR_TIMING: "

/* Z-curve ordering of boxes disabled (leads to insane costs for neighbor box search) */
/*#define BOX_SFC*/

/* fast skip format for box numbers disabled (small (<1%) and ambiguous effects on runtime) */
/*#define BOX_SKIP_FORMAT*/

#define DO_TIMING_SYNC

#ifdef DO_TIMING
# define TIMING_DECL(_decl_)       _decl_
# define TIMING_CMD(_cmd_)         Z_MOP(_cmd_)
#else
# define TIMING_DECL(_decl_)
# define TIMING_CMD(_cmd_)         Z_NOP()
#endif
#ifdef DO_TIMING_SYNC
# define TIMING_SYNC(_c_)          TIMING_CMD(MPI_Barrier(_c_);)
#else
# define TIMING_SYNC(_c_)          Z_NOP()
#endif
#define TIMING_START(_t_)          TIMING_CMD(((_t_) = MPI_Wtime());)
#define TIMING_STOP(_t_)           TIMING_CMD(((_t_) = MPI_Wtime() - (_t_));)
#define TIMING_STOP_ADD(_t_, _r_)  TIMING_CMD(((_r_) += MPI_Wtime() - (_t_));)

#define BOX_BITS                         21
#define BOX_CONST(_b_)                   (_b_##LL)
#define BOX_MASK                         ((BOX_CONST(1) << BOX_BITS) - BOX_CONST(1))

#ifdef BOX_SFC
# define BOX_GET_X(_b_, _x_)             sfc_BOX_GET_X(_b_, _x_)
# define BOX_SET(_v0_, _v1_, _v2_)       sfc_BOX_SET(_v0_, _v1_, _v2_)
# define BOX_ADD(_b_, _a0_, _a1_, _a2_)  BOX_SET(BOX_GET_X((_b_), 0) + (_a0_), BOX_GET_X((_b_), 1) + (_a1_), BOX_GET_X((_b_), 2) + (_a2_))
#else
# define BOX_GET_X(_b_, _x_)             (((_b_) >> ((_x_) * BOX_BITS)) & BOX_MASK)
# define BOX_SET(_v0_, _v1_, _v2_)       ((((box_t) (_v0_)) << (0 * BOX_BITS))|(((box_t) (_v1_)) << (1 * BOX_BITS))|(((box_t) (_v2_)) << (2 * BOX_BITS)))
# define BOX_ADD(_b_, _a0_, _a1_, _a2_)  BOX_SET(BOX_GET_X((_b_), 0) + (_a0_), BOX_GET_X((_b_), 1) + (_a1_), BOX_GET_X((_b_), 2) + (_a2_))
#endif

typedef struct _fcs_near_compute_context_t
{
  fcs_float cutoff;
  const void *compute_param;
  MPI_Comm comm;

  fcs_int running;
#if FCS_NEAR_ASYNC
  fcs_int async;
  pthread_t thread;
#endif

#if FCS_NEAR_OCL
  fcs_ocl_context_t ocl;
#endif /* FCS_NEAR_OCL */

#ifdef DO_TIMING
  double t[10];
#endif

  int comm_size, comm_rank;

  fcs_int periodicity[3];
  box_t *real_boxes, *ghost_boxes;

} fcs_near_compute_context_t;

#ifdef __cplusplus
}
#endif

#endif /* __NEAR_COMMON_H__ */