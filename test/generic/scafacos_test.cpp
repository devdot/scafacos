/*
  Copyright (C) 2011,2012 Olaf Lenz, Michael Hofmann
  
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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <iostream>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <unistd.h>
#include <getopt.h>
#include <mpi.h>

#include "fcs.h"
#include "common/fcs-common/FCSCommon.h"
#include "common/gridsort/gridsort.h"

#include "common.hpp"
#include "Testcase.hpp"
#include "Integration.hpp"


using namespace std;


MPI_Comm communicator;
int comm_rank, comm_size;


static void usage(char** argv, int argc, int c) {
  cout << "Call: " << argv[0];
  for (int i = 1; i < argc; ++i)
    cout << " " << argv[i];
  cout << endl;
  cout << "Usage: " << argv[0] << " [OPTIONS] METHOD FILE" << endl;
  cout << "  OPTIONS:" << endl;
  cout << "    -o | --output <file>" << endl
       << "      write new testcase file <file> using references from the used method" << endl;
  cout << "    -b | --binary" << endl
       << "      write particle data in a machine-dependent binary format to a separate file" << endl
       << "      (only useful together with the option -o <file>, binary output file is" << endl
       << "      <file> with extension '.bin')" << endl;
  cout << "    -p | --portable" << endl
       << "      write particle data in a portable format to a separate file" << endl
       << "      (only usefultogether with the option -o <file>, portable output file is" << endl
       << "      <file> with extension '.dat')" << endl;
  cout << "    -d | --duplication <dup>" << endl
       << "      duplicate a given periodic system in each periodic dimension, <dup> can be" << endl
       << "      a single value X or three values XxYxZ (i.e., one for each dimension)," << endl
       << "      default value <dup>=1 is equivalent to no duplication" << endl;
  cout << "    -k | --keep-duplication" << endl
       << "      keep duplication/generation information in the new testcase file" << endl
       << "      (only useful together with the option -o)" << endl;
  cout << "    -m | --decomposition <mode>" << endl
       << "      use particle decomposition <mode>=atomistic|all_on_master|random|domain" << endl
       << "      (overrides file settings)" << endl;
  cout << "    -C | --cartestian <cart>" << endl
       << "      use Cartesian communicator, <cart> has to be a grid size Xx[Yx[Z]]" << endl
       << "      where X, Y, and Z is either greater 0 or 0 for automatic selection" << endl;
  cout << "    -c | --configuration <conf>" << endl
       << "      use <conf> as configuration string for setting method parameters" << endl;
  cout << "    -i | --iterations <it>" << endl
       << "      perform <it> number of runs with each configuration" << endl;
  cout << "    -r | --results <res>" << endl
       << "      compute results only for <res> of the given particles, <res> can be an" << endl
       << "      absolute number of particles (integer value without '.') or a relative" << endl
       << "      number of the given particles (fractional value with '.')," << endl
       << "      default value <res>=1.0 is equivalent to all particles" << endl;
  cout << "    -u | --compute <comp>" << endl
       << "      compute only <comp> results where <comp> can be '[no]field', '[no]pot'," << endl
       << "      'all', or 'none', default value is <comp>=all" << endl;
  cout << "    -t | --integration <steps>" << endl
       << "      use integration with <steps> time steps (i.e., run METHOD <steps>+1 times)" << endl;
  cout << "    -g | --integ-configuration <conf>" << endl
       << "      use <conf> as configuration string for the integration" << endl;
  cout << "    -s | --resort" << endl
       << "      utilize resort support (if available) to retain the solver specific" << endl
       << "      particle order (i.e., no back sorting), exploit the limited particle" << endl
       << "      movement for integration runs (using -t ...)" << endl;
  cout << "    -a | --allocation <a>" << endl
       << "      control the size of particle data arrays for resort support, <a> can be the" << endl
       << "      minimum size either as absolute value (integer w/o '.') or as fractional" << endl
       << "      number (w/ '.') relative to average size, with prefix '+' <a> can be the" << endl
       << "      additional size either as absolute value (integer w/o '.') or as fractional" << endl
       << "      number (w/ '.') relative to the given particles, minimum and additional" << endl
       << "      size can be set independently, default is minimum size <a>=0 and additional" << endl
       << "      size <a>=+0.1, i.e. no minimum size and 10% additional size" << endl;
  cout << "  METHOD:";
#ifdef FCS_ENABLE_DIRECT
  cout << " direct";
#endif
#ifdef FCS_ENABLE_EWALD
  cout << " ewald";
#endif
#ifdef FCS_ENABLE_FMM
  cout << " fmm";
#endif
#ifdef FCS_ENABLE_MEMD
  cout << " memd";
#endif
#ifdef FCS_ENABLE_MMM1D
  cout << " mmm1d";
#endif
#ifdef FCS_ENABLE_MMM2D
  cout << " mmm2d";
#endif
#ifdef FCS_ENABLE_P2NFFT
  cout << " p2nfft";
#endif
#ifdef FCS_ENABLE_P3M
  cout << " p3m";
#endif
#ifdef FCS_ENABLE_PEPC
  cout << " pepc";
#endif
#ifdef FCS_ENABLE_PP3MG
  cout << " pp3mg";
#endif
#ifdef FCS_ENABLE_VMG
  cout << " vmg";
#endif
  cout << " OR none" << endl;
  cout << "  FILE: XML file describing the system configuration";
#ifdef HAVE_ZLIB_H
  cout << " (can be gzipped!)";
#endif
  cout << endl;

  exit(2);
}

// current configuration
Configuration *current_config;

static struct {
  char infilename[MAX_FILENAME_LENGTH];

  bool have_outfile, have_binfile, have_portable_file;
  char outfilename[MAX_FILENAME_LENGTH], binfilename[MAX_FILENAME_LENGTH], portable_filename[MAX_FILENAME_LENGTH];
  bool keep_dupgen;

  fcs_int periodic_duplications[3];
  fcs_int decomposition;
  bool abort_on_error;

  bool have_method;

  // Which method
  char method[MAX_METHOD_LENGTH];

  // Cartesian communicator
  int cart_comm, cart_dims[3];

  // Configuration string
  char conf[MAX_CONF_LENGTH];
  
  // Number of iterations
  fcs_int iterations;

  // For how many particles results should be computed (0.1 = 10%).
  fcs_float result_particles;

  // Whether to compute field and potential values or not
  bool compute_field, compute_potentials;

  // Utilize resort support of solvers (if available)
  bool resort;

  // Allocation
  fcs_float minalloc, overalloc;

  // Integrate or not, number of time steps (t steps = t+1 computations), and integration configuration string
  bool integrate;
  fcs_int time_steps;
  char integration_conf[MAX_CONF_LENGTH];

} global_params = { 
  "",          /* input file */
  false,       /* output */
  false,       /* binary output */
  false,       /* portable output */
  "",          /* output file */
  "",          /* binary output file */
  "",          /* portable output file */
  false,       /* keep duplication */
  {1, 1, 1},   /* duplication */
  -1,          /* decomposition */
  true,        /* abort on error */
  false,       /* method given */
  "none",      /* method */
  0,           /* cartesian communicator dimensions */
  { 0, 0, 0 }, /* cartesian communicator sizes */
  "",          /* method configuration string */
  1,           /* iterations */
  -1.0,        /* number of particle results */
  true,        /* compute potential values */
  true,        /* compute field values */
  false,       /* do resort */
  0,           /* minimum number of particles to allocate */
  -0.1,        /* additional number of particles to allocate */
  false,       /* integration */
  0,           /* integration time steps */
  ""           /* integration configuration string */
};

static bool check_result(FCSResult result, bool force_abort = false) {
  if (result) {
    cout << "ERROR: Caught error on task " << comm_rank << "!" << endl;
    fcs_result_print_result(result);
    fcs_result_destroy(result);
    if (force_abort || global_params.abort_on_error)
      MPI_Abort(MPI_COMM_WORLD, 1);
    else
      cout << "WARNING: Continuing after error!" << endl;
    return false;
  }
  return true;
}


// split basename and extension of a filename
static void filename_split(char* dest, char* suffix, size_t n, const char* filename)  {
    const char *s = strrchr(filename, '/');
    if (s == NULL) s = filename;
    s = strchr(s, '.');
    snprintf(dest, n, "%.*s", (int) (strlen(filename) - ((s)?strlen(s):0)), filename);
    
    if ( NULL != suffix) snprintf(suffix, n, "%s", s+1);
}


#define STRCMP_FRONT(_s_, _t_)          strncmp((_s_), (_t_), z_min(strlen(_s_), strlen(_t_)))
#define STRCMP_FRONT_IS_EQUAL(_s_, _t_) (STRCMP_FRONT(_s_, _t_) == 0)


// Command line parsing on master
static void parse_commandline(int argc, char* argv[]) {
  char dup0[32], *dup1 = NULL, *dup2 = NULL;

  global_params.conf[0] = '\0';

  static struct option long_options[] =
  {
    {"output",              required_argument, NULL, 'o'},
    {"binary",              no_argument,       NULL, 'b'},
    {"portable",            no_argument,       NULL, 'p'},
    {"duplication",         required_argument, NULL, 'd'},
    {"keep-duplication",    no_argument,       NULL, 'k'},
    {"decomposition",       required_argument, NULL, 'm'},
    {"cartesian",           required_argument, NULL, 'C'},
    {"configuration",       required_argument, NULL, 'c'},
    {"iterations",          required_argument, NULL, 'i'},
    {"results",             required_argument, NULL, 'r'},
    {"compute",             required_argument, NULL, 'u'},
    {"integration",         required_argument, NULL, 't'},
    {"integ-configuration", required_argument, NULL, 'g'},
    {"resort",              no_argument,       NULL, 's'},
    {"allocation",          required_argument, NULL, 'a'},
    {0, 0, 0, 0}
  };

  while (1)
  {
    int option_index = 0;

    int c = getopt_long(argc, argv, "o:bpd:km:C:c:i:r:u:t:g:sa:", long_options, &option_index);

    if (c == -1) break;

    switch (c)
    {
      case 'o':
        strncpy(global_params.outfilename, optarg, MAX_FILENAME_LENGTH);
        global_params.have_outfile = true;
        break;
      case 'b':
        global_params.have_binfile = true;
        break;
      case 'p':
        global_params.have_portable_file = true;
        break;
      case 'd':
        strncpy(dup0, optarg, sizeof(dup0));
        if ((dup1 = strchr(dup0, 'x')))
        {
          *dup1 = 0; ++dup1;
          if ((dup2 = strchr(dup1, 'x')))
          {
            *dup2 = 0; ++dup2;
          }
        }
        global_params.periodic_duplications[0] = global_params.periodic_duplications[1] = global_params.periodic_duplications[2] = (strlen(dup0) > 0)?atoi(dup0):1;
        if (dup1)
        {
          global_params.periodic_duplications[1] = (strlen(dup1) > 0)?atoi(dup1):1;
          global_params.periodic_duplications[2] = (dup2 && strlen(dup2) > 0)?atoi(dup2):1;
        }
        break;
      case 'k':
        global_params.keep_dupgen = true;
        break;
      case 'm':
        if (STRCMP_FRONT("all_on_master", optarg) == 0 || STRCMP_FRONT("master", optarg) == 0)
          global_params.decomposition = DECOMPOSE_ALL_ON_MASTER;
        else if (STRCMP_FRONT("almost_all_on_master", optarg) == 0 || STRCMP_FRONT("almost", optarg) == 0)
          global_params.decomposition = DECOMPOSE_ALMOST_ALL_ON_MASTER;
        else if (STRCMP_FRONT("atomistic", optarg) == 0)
          global_params.decomposition = DECOMPOSE_ATOMISTIC;
        else if (STRCMP_FRONT("random", optarg) == 0)
          global_params.decomposition = DECOMPOSE_RANDOM;
        else if (STRCMP_FRONT("domain", optarg) == 0)
          global_params.decomposition = DECOMPOSE_DOMAIN;
        else if (STRCMP_FRONT("randeq", optarg) == 0)
          global_params.decomposition = DECOMPOSE_RANDOM_EQUAL;
        else
          cout << "WARNING: ignoring unknown decomposition mode '" << optarg << "'" << endl;
        break;
      case 'C':
        strncpy(dup0, optarg, sizeof(dup0));
        if ((dup1 = strchr(dup0, 'x')))
        {
          *dup1 = 0; ++dup1;
          if ((dup2 = strchr(dup1, 'x')))
          {
            *dup2 = 0; ++dup2;
          }
        }
        global_params.cart_comm = 1;
        global_params.cart_dims[0] = (strlen(dup0) > 0)?atoi(dup0):0;
        if (dup1)
        {
          ++global_params.cart_comm;
          global_params.cart_dims[1] = (strlen(dup1) > 0)?atoi(dup1):0;
          if (dup2)
          {
            ++global_params.cart_comm;
            global_params.cart_dims[2] = (strlen(dup2) > 0)?atoi(dup2):0;
          }
        }
        break;
      case 'c':
        if (global_params.conf[0] != '\0') strncat(global_params.conf, ",", MAX_CONF_LENGTH);
        strncat(global_params.conf, optarg, MAX_CONF_LENGTH);
        break;
      case 'i':
        global_params.iterations = atoi(optarg);
        break;
      case 'r':
        global_params.result_particles = fabs(atof(optarg));
        if (strchr(optarg, '.')) global_params.result_particles *= -1;
        break;
      case 'u':
        if (STRCMP_FRONT_IS_EQUAL(optarg, "field")) global_params.compute_field = true;
        else if (STRCMP_FRONT_IS_EQUAL(optarg, "nofield")) global_params.compute_field = false;
        else if (STRCMP_FRONT_IS_EQUAL(optarg, "pot")) global_params.compute_potentials = true;
        else if (STRCMP_FRONT_IS_EQUAL(optarg, "nopot")) global_params.compute_potentials = false;
        else if (STRCMP_FRONT_IS_EQUAL(optarg, "all"))
        {
          global_params.compute_field = true;
          global_params.compute_potentials = true;

        } else if (STRCMP_FRONT_IS_EQUAL(optarg, "none"))
        {
          global_params.compute_field = false;
          global_params.compute_potentials = false;

        } else cout << "WARNING: ignoring unknown compute request '" << optarg << "'" << endl;
        break;
      case 't':
        global_params.integrate = true;
        global_params.time_steps = atoi(optarg);
        break;
      case 'g':
        strncpy(global_params.integration_conf, optarg, MAX_CONF_LENGTH);
        break;
      case 's':
        global_params.resort = true;
        break;
      case 'a':
        if (optarg[0] == '+')
        {
          global_params.overalloc = fabs(atof(optarg + 1));
          if (strchr(optarg + 1, '.')) global_params.overalloc *= -1;

        } else
        {
          global_params.minalloc = fabs(atof(optarg));
          if (strchr(optarg, '.')) global_params.minalloc *= -1;
        }
        break;
      default:
        usage(argv, argc, c);
    }
  }

  if (global_params.have_outfile && (global_params.have_binfile || global_params.have_portable_file))
  {
    // determine basename of output filename and append .bin suffix to create binary filename
    char filename_noext[MAX_FILENAME_LENGTH];
    
    filename_split(filename_noext, NULL, MAX_FILENAME_LENGTH, global_params.outfilename);
    snprintf(global_params.binfilename, MAX_FILENAME_LENGTH, "%s.bin", filename_noext);
    snprintf(global_params.portable_filename, MAX_FILENAME_LENGTH, "%s.dat", filename_noext);
  }

  if (optind >= argc-1) usage(argv, argc, '-');
  strncpy(global_params.method, argv[optind], MAX_METHOD_LENGTH);
  strncpy(global_params.infilename, argv[optind+1], MAX_FILENAME_LENGTH);
}

// broadcast global parameters
static void broadcast_global_parameters() {
  MPI_Bcast(&global_params, sizeof(global_params), MPI_BYTE, MASTER_RANK, communicator);
}


typedef struct
{
  fcs_int total_nparticles, nparticles, max_nparticles;
  fcs_float *positions, *charges, *field, *potentials;

  fcs_float *reference_field, *reference_potentials;

  fcs_int *shuffles;

  fcs_int total_in_nparticles, in_nparticles;
  fcs_float *in_positions, *in_charges;

} particles_t;


static void prepare_particles(particles_t *parts)
{
  fcs_int result_nparticles, total_result_nparticles;

  if (global_params.result_particles < 0) total_result_nparticles = round((fcs_float) current_config->decomp_total_nparticles * -global_params.result_particles);
  else total_result_nparticles = global_params.result_particles;
  
  if (total_result_nparticles > current_config->decomp_total_nparticles) total_result_nparticles = current_config->decomp_total_nparticles;

  fcs_int decomp_prefix = 0;
  MPI_Exscan(&current_config->decomp_nparticles, &decomp_prefix, 1, FCS_MPI_INT, MPI_SUM, communicator);

  result_nparticles = round((fcs_float) total_result_nparticles * (decomp_prefix + current_config->decomp_nparticles) / (fcs_float) current_config->decomp_total_nparticles)
                    - round((fcs_float) total_result_nparticles * decomp_prefix / (fcs_float) current_config->decomp_total_nparticles);

  parts->shuffles = 0;

  if (result_nparticles < current_config->decomp_nparticles)
  {
    parts->shuffles = new fcs_int[result_nparticles];

    for (fcs_int i = 0; i < result_nparticles; ++i)
    {
      parts->shuffles[i] = random() % (current_config->decomp_nparticles - i);

      swap<fcs_float, 3>(current_config->decomp_positions + 3 * i, current_config->decomp_positions + 3 * parts->shuffles[i]);
      swap<fcs_float, 1>(current_config->decomp_charges + i, current_config->decomp_charges + parts->shuffles[i]);
      swap<fcs_float, 3>(current_config->decomp_field + 3 * i, current_config->decomp_field + 3 * parts->shuffles[i]);
      swap<fcs_float, 1>(current_config->decomp_potentials + i, current_config->decomp_potentials + parts->shuffles[i]);
    }
  }

  parts->total_nparticles = total_result_nparticles;
  parts->nparticles = result_nparticles;

  parts->max_nparticles = current_config->decomp_max_nparticles;
  parts->positions = current_config->decomp_positions;
  parts->charges = current_config->decomp_charges;
  parts->field = current_config->decomp_field;
  parts->potentials = current_config->decomp_potentials;
  
  parts->reference_field = current_config->reference_field;
  parts->reference_potentials = current_config->reference_potentials;

  parts->total_in_nparticles = current_config->decomp_total_nparticles - parts->total_nparticles;
  parts->in_nparticles = current_config->decomp_nparticles - parts->nparticles;

  parts->in_positions = parts->positions + 3 * parts->nparticles;
  parts->in_charges = parts->charges + parts->nparticles;

/*  cout << "IN: " << parts->total_nparticles << " / " << parts->total_in_nparticles << endl;
  for (fcs_int i = 0; i < current_config->decomp_nparticles; ++i)
  {
    cout << i << ":"
         << "  " << current_config->decomp_positions[3 * i + 0] << ", " << current_config->decomp_positions[3 * i + 1] << ", " << current_config->decomp_positions[3 * i + 2]
         << "  " << current_config->decomp_charges[i]
         << "  " << current_config->decomp_field[3 * i + 0] << ", " << current_config->decomp_field[3 * i + 1] << ", " << current_config->decomp_field[3 * i + 2]
         << "  " << current_config->decomp_potentials[i]
         << endl;
  }*/
}

static void unprepare_particles(particles_t *parts)
{
  current_config->decomp_nparticles = parts->nparticles;

/*  cout << "OUT: " << parts->total_nparticles << " / " << parts->total_in_nparticles << endl;
  for (fcs_int i = 0; i < current_config->decomp_nparticles; ++i)
  {
    cout << i << ":"
         << "  " << current_config->decomp_positions[3 * i + 0] << ", " << current_config->decomp_positions[3 * i + 1] << ", " << current_config->decomp_positions[3 * i + 2]
         << "  " << current_config->decomp_charges[i]
         << "  " << current_config->decomp_field[3 * i + 0] << ", " << current_config->decomp_field[3 * i + 1] << ", " << current_config->decomp_field[3 * i + 2]
         << "  " << current_config->decomp_potentials[i]
         << endl;
  }*/

  if (parts->shuffles)
  {
    fcs_int result_nparticles = parts->nparticles;

    for (fcs_int i = result_nparticles - 1; i >= 0; --i)
    {
      swap<fcs_float, 3>(current_config->decomp_positions + 3 * parts->shuffles[i], current_config->decomp_positions + 3 * i);
      swap<fcs_float, 1>(current_config->decomp_charges + parts->shuffles[i], current_config->decomp_charges + i);
      swap<fcs_float, 3>(current_config->decomp_field + 3 * parts->shuffles[i], current_config->decomp_field + 3 * i);
      swap<fcs_float, 1>(current_config->decomp_potentials + parts->shuffles[i], current_config->decomp_potentials + i);
    }

    delete[] parts->shuffles;
    parts->shuffles = 0;
  }
}

static double determine_total_energy(fcs_int nparticles, fcs_float *charges, fcs_float *potentials)
{
  fcs_float local, total;
  

  local = 0;

  for (fcs_int i = 0; i < nparticles; i++) local += 0.5 * charges[i] * potentials[i];

  // Compute total energy
  MPI_Allreduce(&local, &total, 1, FCS_MPI_FLOAT, MPI_SUM, communicator);

  return total;
}

#define BACKUP_POSITIONS
#define BACKUP_CHARGES

/*#define PRINT_PARTICLES*/

#ifdef PRINT_PARTICLES
static void print_particles(fcs_int nparticles, fcs_float *positions, fcs_float *charges, fcs_float *field, fcs_float *potentials)
{
  for (fcs_int i = 0; i < nparticles; ++i) printf(" %" FCS_LMOD_INT "d: [%f %f %f] [%f] [%f %f %f] [%f]\n",
    i, positions[3 * i + 0], positions[3 * i + 1], positions[3 * i + 2], charges[i], field[3 * i + 0], field[3 * i + 1], field[3 * i + 2], potentials[i]);
}
#endif

static void run_method(FCS fcs, particles_t *parts)
{
#ifdef BACKUP_POSITIONS
  fcs_float *original_positions;
#endif
#ifdef BACKUP_CHARGES
  fcs_float *original_charges;
#endif
  FCSResult result;

  double t, run_time_sum;

  fcs_int resort_availability, resort = global_params.resort;


  MASTER(cout << "  Setting basic parameters..." << endl);
  MASTER(cout << "    Total number of particles: " << parts->total_nparticles + parts->total_in_nparticles << " (" << parts->total_in_nparticles << " input-only particles)" << endl);
  result = fcs_set_common(fcs, (fcs_get_near_field_flag(fcs) == 0)?0:1,
    current_config->params.box_a, current_config->params.box_b, current_config->params.box_c, current_config->params.box_origin, current_config->params.periodicity, 
    parts->total_nparticles);
  if (!check_result(result)) return;

  if (resort)
  {
    MASTER(cout << "  Enabling resort support...");
    if (fcs_set_resort(fcs, resort) != FCS_RESULT_SUCCESS) resort = 0;
    MASTER(cout << " " << (resort?"OK":"failed") << endl);
  }

  // Compute dipole moment
//  result = fcs_compute_dipole_correction(fcs, parts->total_nparticles + parts->total_in_nparticles,
  result = fcs_compute_dipole_correction(fcs, parts->nparticles, parts->positions, parts->charges,
    current_config->params.epsilon, current_config->field_correction, &current_config->energy_correction);
  MASTER(cout << "  Dipole correction:" << endl);
  MASTER(cout << "    Field correction: "
         << current_config->field_correction[0] << " " 
         << current_config->field_correction[1] << " " 
         << current_config->field_correction[2] << endl);
  MASTER(cout << "    Energy correction: " << current_config->energy_correction << endl);
  if (!check_result(result)) return;

#ifdef FCS_ENABLE_DIRECT
  if (fcs_get_method(fcs) == FCS_METHOD_DIRECT) fcs_direct_set_in_particles(fcs, parts->in_nparticles, parts->in_positions, parts->in_charges);
#endif

  /* create copies of the original positions and charges, as fcs_tune and fcs_run may modify them */
#ifdef BACKUP_POSITIONS
  original_positions = (fcs_float *) malloc(3 * parts->nparticles * sizeof(fcs_float));
  memcpy(original_positions, parts->positions, 3 * parts->nparticles * sizeof(fcs_float));
#endif
#ifdef BACKUP_CHARGES
  original_charges = (fcs_float *) malloc(parts->nparticles * sizeof(fcs_float));
  memcpy(original_charges, parts->charges, parts->nparticles * sizeof(fcs_float));
#endif

  fcs_set_max_local_particles(fcs, parts->max_nparticles);

  // Tune and time method
  MASTER(cout << "  Tuning method..." << endl);
  MPI_Barrier(communicator);
  t = MPI_Wtime();
  result = fcs_tune(fcs, parts->nparticles, parts->positions, parts->charges);
  MASTER(fcs_print_parameters(fcs));
  if (!check_result(result)) return;
  MPI_Barrier(communicator);
  t = MPI_Wtime() - t;
  MASTER(cout << "    Time: " << scientific << t << endl);
  
  // Run and time method
  MASTER(cout << "  Running method..." << endl);
  DEBUG(cout << comm_rank << ": local number of particles: " << parts->nparticles << "(" << parts->in_nparticles << " input-only)" << endl);

  run_time_sum = 0;

  for (fcs_int i = 0; i < global_params.iterations; ++i)
  {
    /* restore original positions and charges, as fcs_tune and fcs_run may have modified them */
#ifdef BACKUP_POSITIONS
    memcpy(parts->positions, original_positions, 3 * parts->nparticles * sizeof(fcs_float));
#endif
#ifdef BACKUP_CHARGES
    memcpy(parts->charges, original_charges, parts->nparticles * sizeof(fcs_float));
#endif

#ifdef PRINT_PARTICLES
    MASTER(cout << "Particles before fcs_run: " << parts->nparticles << endl);
    print_particles(parts->nparticles, parts->positions, parts->charges,
            parts->field, parts->potentials);
#endif

    MPI_Barrier(communicator);
    t = MPI_Wtime();
    result = fcs_run(fcs, parts->nparticles, parts->positions, parts->charges, parts->field, parts->potentials);
    if (!check_result(result)) return;
    MPI_Barrier(communicator);
    t = MPI_Wtime() - t;
    MASTER(cout << "    #" << i << " time: " << scientific << t << endl);

#ifdef PRINT_PARTICLES
    MASTER(cout << "Particles after fcs_run: " << parts->nparticles << endl);
    print_particles(parts->nparticles, parts->positions, parts->charges,
            parts->field, parts->potentials);
#endif

    run_time_sum += t;
  }

#ifdef BACKUP_POSITIONS
  free(original_positions);
#endif
#ifdef BACKUP_CHARGES
  free(original_charges);
#endif

  MASTER(cout << "    Average time: " << scientific << run_time_sum / global_params.iterations << endl);
  MASTER(cout << "    Total time:   " << scientific << run_time_sum << endl);

  current_config->have_result_values[0] = 1;  // have potentials results
  current_config->have_result_values[1] = 1;  // have field results

  fcs_get_resort_availability(fcs, &resort_availability);
  if (resort_availability)
  {
    fcs_get_resort_particles(fcs, &parts->nparticles);

    MASTER(cout << "    Resorting reference potential and field values..." << endl);

    t = MPI_Wtime();
    if (parts->reference_potentials != NULL)
        fcs_resort_floats(fcs, parts->reference_potentials, NULL, 1);
    if (parts->reference_field != NULL)
        fcs_resort_floats(fcs, parts->reference_field, NULL, 3);
    t = MPI_Wtime() - t;

    MASTER(printf("     = %f second(s)\n", t));

  } else if (resort) MASTER(cout << "    Resorting enabled, but failed!" << endl);

/*  for (fcs_int i = 0; i < parts->nparticles; ++i)
    cout << i << ": " << parts->positions[3 * i + 0] << ", "
                      << parts->positions[3 * i + 1] << ", "
                      << parts->positions[3 * i + 2] << "  " << parts->charges[i] << "  "
                      << parts->field[3 * i + 0] << ", "
                      << parts->field[3 * i + 1] << ", "
                      << parts->field[3 * i + 2] << "  " << parts->potentials[i] << endl;*/

  if (parts->field)
  {
    // apply dipole correction to the fields
    for (fcs_int pid = 0; pid < parts->nparticles; pid++)
    {
      parts->field[3*pid] += current_config->field_correction[0];
      parts->field[3*pid+1] += current_config->field_correction[1];
      parts->field[3*pid+2] += current_config->field_correction[2];
    }
  }
}

static void no_method() {
  current_config->have_result_values[0] = 0;  // no potentials results
  current_config->have_result_values[1] = 0;  // no field results
}

static void run_integration(FCS fcs, particles_t *parts, Testcase *testcase)
{
  integration_t integ;

  fcs_int resort_availability, resort = global_params.resort;

  fcs_float *v_cur, *f_old, *f_cur, e, max_particle_move;
#ifdef BACKUP_POSITIONS
  fcs_float *xyz_old;
#endif
  FCSResult result;
  fcs_int r = 0;

  double t, tune_time_sum, run_time_sum, resort_time_sum;


  MASTER(cout << "  Integration with " << global_params.time_steps << " time step(s) " << (resort?"with":"without") << " utilization of resort support" << endl);

  if (!parts->field)
  {
    MASTER(cout << "  ERROR: Performing integration requires computing of field values!" << endl);
    return;
  }

  v_cur = new fcs_float[3 * parts->max_nparticles];
  f_old = new fcs_float[3 * parts->max_nparticles];
  f_cur = parts->field;
  
#ifdef BACKUP_POSITIONS
  xyz_old = new fcs_float[3 * parts->max_nparticles];
#endif

  /* setup integration parameters */
  integ_setup(&integ, global_params.time_steps, global_params.resort, global_params.integration_conf);

  integ_system_setup(&integ, current_config->params.box_a, current_config->params.box_b, current_config->params.box_c, current_config->params.box_origin, current_config->params.periodicity);

  MASTER(
    integ_print_settings(&integ, "    ");
  );

  /* init integration (incl. velocity and field values) */
  integ_init(&integ, parts->nparticles, v_cur, f_cur);

  if (fcs != FCS_NULL)
  {
    /* init method */
    result = fcs_set_common(fcs, (fcs_get_near_field_flag(fcs) == 0)?0:1,
      current_config->params.box_a, current_config->params.box_b, current_config->params.box_c, current_config->params.box_origin, current_config->params.periodicity, parts->total_nparticles);
    if (!check_result(result)) return;

    if (fcs_set_resort(fcs, resort) != FCS_RESULT_SUCCESS) resort = 0;

  } else
  {
    for (fcs_int i = 0; i < parts->nparticles; ++i) parts->field[3 * i + 0] = parts->field[3 * i + 1] = parts->field[3 * i + 2] = parts->potentials[i] = 0.0;
  }

  MASTER(cout << "  Initial step" << endl);

  tune_time_sum = run_time_sum = resort_time_sum = 0;

  while (1)
  {
    /* store previous field values */
    for (fcs_int i = 0; i < 3 * parts->nparticles; ++i) f_old[i] = f_cur[i];

#ifdef PRINT_PARTICLES
    MASTER(cout << "Particles before fcs_run: " << parts->nparticles << endl);
    print_particles(parts->nparticles, parts->positions, parts->charges, parts->field, parts->potentials);
#endif

    if (fcs != FCS_NULL)
    {
      fcs_set_max_local_particles(fcs, parts->max_nparticles);

      /* tune method */
      MASTER(cout << "    Tune method..." << endl);
      MPI_Barrier(communicator);
      t = MPI_Wtime();
      result = fcs_tune(fcs, parts->nparticles, parts->positions, parts->charges);
      MPI_Barrier(communicator);
      t = MPI_Wtime() - t;
      if (!check_result(result)) return;
      MASTER(printf("     = %f second(s)\n", t));

      tune_time_sum += t;

      /* store old positions */
#ifdef BACKUP_POSITIONS
      memcpy(xyz_old, parts->positions, parts->nparticles * 3 * sizeof(fcs_float));
#endif

      MASTER(cout << "    Run method..." << endl);
      MPI_Barrier(communicator);
      t = MPI_Wtime();
      result = fcs_run(fcs, parts->nparticles, parts->positions, parts->charges, parts->field, parts->potentials);
      MPI_Barrier(communicator);
      t = MPI_Wtime() - t;
      if (!check_result(result)) return;
      MASTER(printf("     = %f second(s)\n", t));

      run_time_sum += t;

      /* restore old positions */
#ifdef BACKUP_POSITIONS
      memcpy(parts->positions, xyz_old, parts->nparticles * 3 * sizeof(fcs_float));
#endif
    }

    fcs_get_resort_availability(fcs, &resort_availability);
    if (resort_availability)
    {
      fcs_get_resort_particles(fcs, &parts->nparticles);

      MASTER(cout << "    Resorting old velocity and field values..." << endl);

      MPI_Barrier(communicator);
      t = MPI_Wtime();
      fcs_resort_floats(fcs, v_cur, NULL, 3);
      fcs_resort_floats(fcs, f_old, NULL, 3);
      MPI_Barrier(communicator);
      t = MPI_Wtime() - t;
      MASTER(printf("     = %f second(s)\n", t));

      resort_time_sum += t;

      MASTER(cout << "    Resorting reference potential and field values..." << endl);

      t = MPI_Wtime();
      if (parts->reference_potentials != NULL) fcs_resort_floats(fcs, parts->reference_potentials, NULL, 1);
      if (parts->reference_field != NULL) fcs_resort_floats(fcs, parts->reference_field, NULL, 3);
      t = MPI_Wtime() - t;

      MASTER(printf("     = %f second(s)\n", t));

      /* resort the restored old positions */
#ifdef BACKUP_POSITIONS
      fcs_resort_floats(fcs, parts->positions, NULL, 3);
#endif

    } else if (resort) MASTER(cout << "    Resorting enabled, but failed!" << endl);

#ifdef PRINT_PARTICLES
    MASTER(cout << "Particles after fcs_run: " << parts->nparticles << endl);
    print_particles(parts->nparticles, parts->positions, parts->charges, parts->field, parts->potentials);
#endif

    if (0 != integ.output_steps && global_params.have_outfile) {
      char outfilename[MAX_FILENAME_LENGTH];
      char binfilename[MAX_FILENAME_LENGTH];
      char portable_filename[MAX_FILENAME_LENGTH];

      char filename_noext[MAX_FILENAME_LENGTH];
      char filename_suffix[MAX_FILENAME_LENGTH];

      filename_split(filename_noext, filename_suffix, MAX_FILENAME_LENGTH, global_params.outfilename);
      snprintf(outfilename, MAX_FILENAME_LENGTH, "%s_%08" FCS_LMOD_INT "d.%s", filename_noext, r, filename_suffix);

      // delete existing binary file
      if (global_params.have_binfile) {
        filename_split(filename_noext, filename_suffix, MAX_FILENAME_LENGTH, global_params.binfilename);
        snprintf(binfilename, MAX_FILENAME_LENGTH, "%s_%08" FCS_LMOD_INT "d.%s", filename_noext, r, filename_suffix);
        MPI_File_delete(binfilename, MPI_INFO_NULL);
      }

      // delete existing portable file
      if (global_params.have_portable_file) {
        filename_split(filename_noext, filename_suffix, MAX_FILENAME_LENGTH, global_params.portable_filename);
        snprintf(portable_filename, MAX_FILENAME_LENGTH, "%s_%08" FCS_LMOD_INT "d.%s", filename_noext, r, filename_suffix);
        MPI_File_delete(portable_filename, MPI_INFO_NULL);
      }  

      // called by all processes so that we can write out all particles of all processes
      testcase->write_file(outfilename,
        global_params.have_binfile?binfilename:NULL,
        global_params.have_portable_file?portable_filename:NULL,
        global_params.keep_dupgen);
    }

    if (r > 0)
    {
      MASTER(cout << "    Update velocities..." << endl);
      integ_update_velocities(&integ, parts->nparticles, v_cur, f_old, f_cur, parts->charges);
    }

    MASTER(cout << "    Determine total energy..." << endl);
    e = determine_total_energy(parts->nparticles, parts->charges, parts->potentials);
    MASTER(cout << "      total energy = " << scientific << e << endl);

    if (r >= global_params.time_steps) break;

    ++r;

    MASTER(cout << "  Time-step #" << r << endl);

    MASTER(cout << "    Update positions..." << endl);
    integ_update_positions(&integ, parts->nparticles, parts->positions, NULL, v_cur, f_cur, parts->charges, &max_particle_move);

    if (resort_availability && integ.max_move)
    {
      MASTER(cout << "    Set max particle move = " << max_particle_move);

      result = fcs_set_max_particle_move(fcs, max_particle_move);

      if (!result) MASTER(cout << endl);
      else
      {
        MASTER(cout << " (failed because not supported!)" << endl);
        fcs_result_destroy(result);
      }
    }

    integ_correct_positions(&integ, parts->nparticles, parts->positions);
    fcs_set_box_a(fcs, integ.box_a);
    fcs_set_box_b(fcs, integ.box_b);
    fcs_set_box_c(fcs, integ.box_c);
    fcs_set_box_origin(fcs, integ.box_origin);
  }

  current_config->have_reference_values[0] = 0;
  current_config->have_reference_values[1] = 0;
  current_config->have_result_values[0] = 1;
  current_config->have_result_values[1] = 1;

  MASTER(cout << "  Tune time:   " << scientific << tune_time_sum << endl);
  MASTER(cout << "  Run time:    " << scientific << run_time_sum << endl);
  MASTER(cout << "  Resort time: " << scientific << resort_time_sum << endl);

  delete[] v_cur;
  delete[] f_old;

#ifdef BACKUP_POSITIONS
  delete[] xyz_old;
#endif
}

int main(int argc, char* argv[])
{
  Testcase *testcase = 0;

#if FCS_ENABLE_PEPC
  int mpi_thread_requested = MPI_THREAD_MULTIPLE;
  int mpi_thread_provided;

  MPI_Init_thread(&argc, &argv, mpi_thread_requested, &mpi_thread_provided);
#else
  MPI_Init(&argc, &argv);
#endif

  communicator = MPI_COMM_WORLD;

  MPI_Comm_rank(communicator,&comm_rank);
  MPI_Comm_size(communicator,&comm_size);

#if FCS_ENABLE_PEPC
  if (mpi_thread_provided < mpi_thread_requested) {
    MASTER(printf("Call to MPI_INIT_THREAD failed.\n" \
                  "Requested/provided level of multithreading: %d / %d.\n" \
                  "Continuing but expect program crash.\n", \
                  mpi_thread_requested, mpi_thread_provided));
  }
#endif

  MASTER(cout << "Running generic test with " << comm_size << " processes" << endl);

  if (comm_rank == MASTER_RANK) {
    parse_commandline(argc, argv);

    if (strcasecmp(global_params.method, "none") == 0) global_params.have_method = false;
    else global_params.have_method = true;
  }

  srandom((comm_rank + 1) * 2501);

  global_params.abort_on_error = false;

  broadcast_global_parameters();

  MASTER(
    if (global_params.cart_comm > 0)
    {
      cout << "Trying to use " << global_params.cart_comm << "d Cartesian communicators of size " << global_params.cart_dims[0];
      if (global_params.cart_comm > 1) cout << "x" << global_params.cart_dims[1];
      if (global_params.cart_comm > 2) cout << "x" << global_params.cart_dims[2];
      cout << "!" << endl;
    }
  );

  if (global_params.have_outfile && global_params.resort)
  {
    MASTER(cout << "Disabling resort support (-s) when output file should be written!" << endl);
    global_params.resort = false;
  }

  MASTER(cout << "Particle array allocation: minalloc: " << global_params.minalloc << ", overalloc: " << global_params.overalloc << endl);

  testcase = new Testcase();

  if (comm_rank == MASTER_RANK) {
    // Read testcase data on master
    try {
      testcase->read_file(global_params.infilename, global_params.periodic_duplications, global_params.decomposition);
    } catch (ParserError &er) {
      cout << "ERROR: " << er.what() << endl;
      MPI_Abort(communicator, 1);
    }
  }

  if (global_params.have_method)
  {
    testcase->error_field = 1.0;
    testcase->error_potential = 1.0;
    testcase->reference_method = global_params.method;
  }

  testcase->broadcast_config(MASTER_RANK, communicator);

  MASTER(cout << "Config parameters:" << endl);
  const char *xml_method_conf = testcase->get_method_config();
  MASTER(cout << "  XML file: " << xml_method_conf << endl);
  MASTER(cout << "  Command line: " << global_params.conf << endl);

  fcs_int config_count = 0;

  vector<Configuration*>::iterator config;

  if (comm_rank == MASTER_RANK) config = testcase->configurations.begin();

  // Loop over configurations
  while (1) {

    // Check whether we have another configuration
    fcs_int quit_loop = 0;
    if (comm_rank == MASTER_RANK && config == testcase->configurations.end()) quit_loop = 1;
    MPI_Bcast(&quit_loop, 1, FCS_MPI_INT, MASTER_RANK, communicator);

    if (quit_loop) break;

    // Create new configuration on non-root processes
    if (comm_rank != MASTER_RANK)
    {
      Configuration *c = new Configuration();
      testcase->configurations.push_back(c);
      config = testcase->configurations.end();
      --config;
    }
 
    current_config = *config;

    // Broadcast configuration parameters
    current_config->broadcast_config();

    MASTER(cout << "Processing configuration " << config_count << "..." << endl);

    FCS fcs = FCS_NULL;
    FCSResult result;
    MPI_Comm fcs_comm = communicator;

    if (global_params.cart_comm > 0)
    {
      int dims[3], periods[3];

      dims[0] = global_params.cart_dims[0];
      dims[1] = global_params.cart_dims[1];
      dims[2] = global_params.cart_dims[2];

      MPI_Dims_create(comm_size, global_params.cart_comm, dims);

      periods[0] = current_config->params.periodicity[0];
      periods[1] = current_config->params.periodicity[1];
      periods[2] = current_config->params.periodicity[2];

      MASTER(
        cout << "  Creating " << global_params.cart_comm << "d Cartesian communicator of size " << dims[0];
        if (global_params.cart_comm > 1) cout << "x" << dims[1];
        if (global_params.cart_comm > 2) cout << "x" << dims[2];
        cout << " with periodicity (" << periods[0] << "," << periods[1] << "," << periods[2] << ")" << endl;
      );

      MPI_Cart_create(communicator, global_params.cart_comm, dims, periods, 0, &fcs_comm);
    }

    if (global_params.have_method)
    {
      MASTER(cout << "  Initializing FCS, method " << global_params.method << "..." << endl);
      result = fcs_init(&fcs, global_params.method, fcs_comm);
      check_result(result, true);

    } else MASTER(cout << " No method chosen!" << endl);

    MASTER(cout << "  Setting method configuration parameters..." << endl);
    if (strlen(xml_method_conf) > 0)
    {
      result = fcs_set_parameters(fcs, xml_method_conf, FCS_TRUE);
      check_result(result, (fcs_result_get_return_code(result) != FCS_ERROR_WRONG_ARGUMENT));
    }
    if (strlen(global_params.conf) > 0)
    {
      result = fcs_set_parameters(fcs, global_params.conf, FCS_FALSE);
      check_result(result, true);
    }

    // Distribute particles
    current_config->decompose_particles(global_params.compute_field, global_params.compute_potentials, global_params.resort?global_params.minalloc:0, global_params.resort?global_params.overalloc:0);

    particles_t parts;
    prepare_particles(&parts);

    if (global_params.integrate) run_integration(fcs, &parts, testcase);
    else {
      // Run method or do nothing
      if (global_params.have_method) run_method(fcs, &parts);
      else no_method();
    }

    unprepare_particles(&parts);

    // Compute errors
    errors_t err;
    bool have_errors = current_config->compute_errors(&err);

    // output the errors on the master node
    if (comm_rank == MASTER_RANK)
      if (have_errors) print_errors(&err, "  ");

    if (global_params.have_outfile) {
      // Write the computed data as new references into the testcase
      if (current_config->have_result_values[0])
        memcpy(current_config->dup_input_potentials, current_config->decomp_potentials, current_config->dup_input_nparticles*sizeof(fcs_float));
      if (current_config->have_result_values[1])
        memcpy(current_config->dup_input_field, current_config->decomp_field, current_config->dup_input_nparticles*3*sizeof(fcs_float));
    }

    // Free particles
    current_config->free_decomp_particles();

    if (global_params.have_method)
    {
      MASTER(cout << "Destroying FCS ..." << endl);
      result = fcs_destroy(fcs);
      check_result(result, true);
    }

    if (fcs_comm != communicator) MPI_Comm_free(&fcs_comm);

    // proceed to the next configuration
    config++;
    config_count++;
  }

  if (global_params.have_outfile) {
    if (comm_rank == MASTER_RANK)
    {
      cout << "New reference data: method: " << testcase->reference_method
           << ", potential error: " << testcase->error_potential << ", field error: " << testcase->error_field << endl;
    }

    // delete existing binary file
    if (global_params.have_binfile) MPI_File_delete(global_params.binfilename, MPI_INFO_NULL);

    // delete existing portable file
    if (global_params.have_portable_file) MPI_File_delete(global_params.portable_filename, MPI_INFO_NULL);

    // called by all processes so that we can write out all particles of all processes
    testcase->write_file(global_params.outfilename,
      global_params.have_binfile?global_params.binfilename:NULL,
      global_params.have_portable_file?global_params.portable_filename:NULL,
      global_params.keep_dupgen);
  }

  delete testcase;

  MPI_Finalize();

  MASTER(cout << "Done." << endl);
}
