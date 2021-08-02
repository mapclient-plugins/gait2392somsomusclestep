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

from setuptools import setup, find_packages
import io


def readfile(filename, split=False):
    with io.open(filename, encoding="utf-8") as stream:
        if split:
            return stream.read().split("\n")
        return stream.read()


package_readme = readfile("README.rst")[3:]  # skip title
package_license = readfile("LICENSE")
package_dependencies = [
    "setuptools",
    "PySide2",
    "numpy",
    "gias2",
    "scipy"
]

setup(name=u'mapclientplugins.gait2392somsomusclestep',
      version='0.1',
      description='',
      long_description='\n'.join(package_readme) + package_license,
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
