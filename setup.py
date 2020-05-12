# Had to export the following before running
# export CFLAGS=-sysroot,/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.7.sdk
# export LDFLAGS=-L/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.7.sdk/usr/lib
#
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os
import numpy

gco_directory = "../pygco/gco-v3.0"

files = ['GCoptimization.cpp', 'LinkedBlockList.cpp']

files = [os.path.join(gco_directory, f) for f in files]
files.insert(0, "gco_python.pyx")

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension(
        "pygco", files, language="c++",
        include_dirs=[gco_directory, numpy.get_include()],
        library_dirs=[gco_directory],
        extra_compile_args=["-fpermissive", "-Ofast"]
    )]
)
