
lib_LTLIBRARIES =
noinst_LTLIBRARIES =

if ENABLE_SINGLE_LIB
noinst_LTLIBRARIES += libfcs_ocl.la
else
lib_LTLIBRARIES += libfcs_ocl.la
endif

libfcs_ocl_la_CPPFLAGS = -I$(top_srcdir)/lib

libfcs_ocl_la_SOURCES = fcs_ocl.c fcs_ocl.h

nodist_libfcs_ocl_la_SOURCES = fcs_ocl.cl_str.h fcs_math.cl_str.h
EXTRA_DIST = fcs_ocl.cl.h fcs_math.cl.h
BUILT_SOURCES = fcs_ocl.cl_str.h fcs_math.cl_str.h
CLEANFILES = fcs_ocl.cl_str.h fcs_math.cl_str.h

fcs_ocl.cl_str.h: $(srcdir)/fcs_ocl.cl.h Makefile $(top_srcdir)/build-aux/opencl-stringify
	USE_CPP="no" USE_HEX="no" HEXPAD="16" HEXNULL="yes" \
	$(SHELL) $(top_srcdir)/build-aux/opencl-stringify $< > $@

fcs_math.cl_str.h: $(srcdir)/fcs_math.cl.h Makefile $(top_srcdir)/build-aux/opencl-stringify
	USE_CPP="no" USE_HEX="no" HEXPAD="16" HEXNULL="yes" \
	$(SHELL) $(top_srcdir)/build-aux/opencl-stringify $< > $@
