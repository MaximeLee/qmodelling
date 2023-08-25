from setuptools import setup, find_namespace_packages, find_packages
from Cython.Build import cythonize
import os
import numpy

# Recursively find all pyx/pxd files
cython_files = []
for root, _, files in os.walk('qmodelling'):
    for file in files:
        if file.endswith(".pyx") or file.endswith(".pxd"):
            cython_files.append(os.path.join(root, file))

setup(
    ext_modules=cythonize(cython_files),  # Compile all .pyx files in all discovered packages
    include_dirs=[numpy.get_include()],
    script_args=['build_ext', '--inplace']
)
#setup(
#    ext_modules=cythonize(pyx_files),  # Compile all .pyx files in all discovered packages
#    include_dirs=[numpy.get_include()],
#    script_args=['build_ext', '--inplace']
#)

#setup(
#    ext_modules=cythonize('qmodelling/basis/*.pyx'),  # Compile all .pyx files in all discovered packages
#    include_dirs=[numpy.get_include()],
#    script_args=['build_ext', '--inplace']
#)
