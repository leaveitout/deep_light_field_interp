from setuptools import setup

setup(
    name='deep_light_field_interp',
    version='0.1',
    packages=['deeplfinterp', 'deeplfinterp.util', 'deeplfinterp.tools',
              'deeplfinterp.models', 'deeplfinterp.datasets',
              'deeplfinterp.experiments'],
    url='',
    license='',
    author='Seán Bruton',
    author_email='sbruton@tcd.ie',
    description='Light Field Interpolation',
    install_requires=['torch', 'numpy', 'h5py', 'progressbar']
)
