from setuptools import setup
from setuptools import find_packages


setup(name='KFS',
      version='0.0.1',
      description='Keras for Science',
      author='Michael Oliver',
      author_email='michael.d.oliver@gmail.com',
      url='https://github.com/the-moliver/kfs',
      download_url='https://github.com/the-moliver/kfs/tarball/0.0.1',
      license='MIT',
      install_requires=['theano', 'pyyaml', 'six', 'keras'],
      extras_require={
          'h5py': ['h5py'],
      },
      packages=find_packages())
