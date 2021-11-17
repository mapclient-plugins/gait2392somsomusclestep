"""
MAP Client, a program to generate detailed musculoskeletal models for OpenSim.
    Copyright (C) 2012  University of Auckland

This file is part of MAP Client. (http://launchpad.net/mapclient)

    MAP Client is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    MAP Client is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with MAP Client.  If not, see <http://www.gnu.org/licenses/>..
"""
import codecs
import io
import os
import re

from setuptools import setup, find_packages


SETUP_DIR = os.path.dirname(os.path.abspath(__file__))


def read(*parts):
    with codecs.open(os.path.join(SETUP_DIR, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def readfile(filename, split=False):
    with io.open(filename, encoding="utf-8") as stream:
        if split:
            return stream.read().split("\n")
        return stream.read()


package_readme = readfile("README.rst")[3:]  # skip title
package_license = readfile("LICENSE")
package_dependencies = [
    "PySide2",
    "numpy",
    "gias3.musculoskeletal",
    "scipy"
]

setup(
    name=u'mapclientplugins.gait2392somsomusclestep',
    version=find_version('mapclientplugins', 'gait2392somsomusclestep', '__init__.py'),
    description='',
    long_description='\n'.join(package_readme) + package_license,
    long_description_content_type='text/x-rst',
    classifiers=[],
    author=u'Ju Zhang',
    author_email='',
    url='',
    license='APACHE',
    packages=find_packages(exclude=['ez_setup', ]),
    namespace_packages=['mapclientplugins'],
    include_package_data=True,
    zip_safe=False,
    install_requires=package_dependencies,
)
