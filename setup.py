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
    "scipy",
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
      packages=find_packages(exclude=['ez_setup',]),
      namespace_packages=['mapclientplugins'],
      include_package_data=True,
      zip_safe=False,
      install_requires=package_dependencies,
      )
