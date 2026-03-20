import os
import pathlib

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as build_ext_orig


class CMakeExtension(Extension):

    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])


class build_ext(build_ext_orig):

    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        #super().run()

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))

        config = 'Release ' + '--j4'
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + str(extdir.parent.absolute()),
            '-DCMAKE_BUILD_TYPE=' + config
        ]

        build_args = [
            '--config', config
        ]

        os.chdir(str(build_temp))
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            if os.name != "nt":
                self.spawn(['cmake', '--build', '.'] + build_args)
            else:
                self.spawn(['cmake', '--build', '.'])
        os.chdir(str(cwd))

required = ["numpy", "scipy", "matplotlib"]

# The package version number is set in `src/PyPO/__init__.py` only.
# Use bump-my-version to update it and the `README.md` file.
init_file = pathlib.Path(__file__).parent.absolute() / "src/PyPO/__init__.py"
default_v = '0.0.0'
version = default_v # Subtle trick here to avoid bump-my-version finding these lines
with open(init_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if '__version__ =' in line:
            version = line.split('=')[-1].strip().strip("'\"")
            break


setup(
    name='PyPO-pkg',
    license="MIT",
    version=version,
    author="Arend Moerman",
    install_requires = required,
    package_dir = {'': 'src'},
    packages=['PyPO'],
    ext_modules=[CMakeExtension(os.path.join("PyPO", "libs"))],
    cmdclass={'build_ext': build_ext},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
        "Programming Language :: C++",
        "Environment :: GPU :: NVIDIA CUDA :: 11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    python_requires='>=3.8',
)
