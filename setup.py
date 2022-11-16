import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


setup(
    name='videochef',
    version='0.0.3',
    license='MIT',
    description='Whip up some fast, embarrassingly parallel video processing',
    author='Jonah Pearl',
    author_email='jonahpearl@g.harvard.edu',
    url='https://github.com/jonahpearl/videochef',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    project_urls={

        'Issue Tracker': 'https://github.com/jonahpearl/videochef/issues',
    },
    python_requires='>=3.6',
    install_requires=[
        'numpy', 'click', 'av', 'tqdm',
    ],
)