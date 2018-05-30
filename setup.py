from distutils.core import setup
import YSOVAR


setup(name='pYSOVAR',
      version=YSOVAR.__version__,
      author='H. M. Guenther and K. Poppenhaeger',
      url='https://github.com/YSOVAR/YSOVAR',
      packages=['YSOVAR'],
      description='lightcurve analysis for the YSOVAR project',
      requires=['astropy', 'scipy', 'numpy', 'matplotlib'],
      install_requires=['astropy>=0.4', 'scipy', 'numpy', 'matplotlib'],
      author_email='hgunther@mit.edu',
      license='GPL v3',
      long_description=YSOVAR.__doc__,
      download_url='https://github.com/YSOVAR/YSOVAR/tarball/1.0',
      classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2.7',
          'Topic :: Scientific/Engineering :: Astronomy',
      ],
      )
