# Remove files created during configure.
DISTCLEANFILES = fcs-package.info

ACLOCAL_AMFLAGS = -I m4

if ENABLE_LIBRARY_INSTALL
lib_LTLIBRARIES = libfcs_p2nfft.la
else
noinst_LTLIBRARIES = libfcs_p2nfft.la
endif

# Directory of fftw.h
AM_CPPFLAGS = $(fftw3_CPPFLAGS) 

# Directory of pfft.h
AM_CPPFLAGS += $(pfft_CPPFLAGS)

# Directory of pnfft.h
AM_CPPFLAGS += $(pnfft_CPPFLAGS)

# Directory of fcs.h
AM_CPPFLAGS += -I$(top_srcdir)/../../src

# Libraries to add to all programs that are built.

libfcs_p2nfft_la_CPPFLAGS = $(fftw3_CPPFLAGS) $(pfft_CPPFLAGS) $(pnfft_CPPFLAGS) -I$(top_srcdir)/../../src -I$(top_srcdir)/../../lib -I$(top_srcdir)/../../lib/common/fcs-common -I$(top_builddir)/../..
libfcs_p2nfft_la_SOURCES = \
	types.h \
	utils.h \
	constants.h \
	p2nfft.h \
	init.c init.h \
	tune.c tune.h \
	run.c run.h \
	bessel_k.c bessel_k.h \
	part_derive_one_over_norm_x.c part_derive_one_over_norm_x.h \
	parameters.c parameters.h \
	nearfield.c nearfield.h \
	kernels.c kernels.h \
	regularization.c regularization.h \
	taylor2p.c taylor2p.h \
	interpolation.c interpolation.h \
	cg_cos_coeff.c cg_cos_coeff.h \
	cg_cos_err.c cg_cos_err.h \
	cg_cos_coeff_sym.c cg_cos_coeff_sym.h \
	cg_cos_err_sym.c cg_cos_err_sym.h

if ENABLE_OPENCL
nodist_libfcs_p2nfft_la_SOURCES = nearfield.cl_str.h
BUILT_SOURCES = nearfield.cl_str.h
CLEANFILES = nearfield.cl_str.h

nearfield.cl_str.h: $(srcdir)/nearfield.cl.c Makefile $(top_srcdir)/build-aux/opencl-stringify
	USE_CPP="yes" CPP="$(CPP)" CPPFLAGS="$(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) $(libfcs_p2nfft_la_CPPFLAGS) $(CPPFLAGS)" FILTER_TO="HERE_COMES_THE_CODE" USE_HEX="yes" HEXPAD="16" HEXNULL="yes" \
	$(SHELL) $(top_srcdir)/build-aux/opencl-stringify $< > $@
endif

#noinst_PROGRAMS = tests/test_random
#tests_test_random_SOURCES = tests/test_random.c
#tests_test_random_LDADD = libfcs_p2nfft.a \
#                          ../../src/libfcs.a \
#                          ../common/pnfft/.libs/libpnfft.a \
#                          ../common/pfft/.libs/libpfft.a \
#                          ../direct/libdirect.a \
#                          ../common/near/libnear.a
