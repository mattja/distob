from setuptools import setup
from setuptools.command.test import test as TestCommand
import sys
import codecs
import os
import re

here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    return codecs.open(os.path.join(here, *parts), 'r').read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

class Tox(TestCommand):
    user_options = [('tox-args=', 'a', "Arguments to pass to tox")]
    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.tox_args = None
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True
    def run_tests(self):
        import tox
        import shlex
        errno = tox.cmdline(args=shlex.split(self.tox_args))
        sys.exit(errno)

setup(
    name='distob',
    version=find_version('distob', '__init__.py'),
    url='http://github.com/mattja/distob/',
    license='GPLv3+',
    author='Matthew J. Aburn',
    install_requires=['IPython>=2.1',
                      'pyzmq>=2.1.11',
                      'dill>=0.2.1'],
    tests_require=['tox'],
    cmdclass = {'test': Tox},
    author_email='mattja6@gmail.com',
    description='Distributed computing made easier, using remote objects',
    long_description=read('README.md'),
    packages=['distob'],
    platforms='any',
    zip_safe=False,
    classifiers = [
        'Programming Language :: Python',
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: System :: Distributed Computing',
        ],
    extras_require={'remote_arrays': ['numpy>=1.6']}
)
