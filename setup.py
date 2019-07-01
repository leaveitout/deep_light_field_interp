from setuptools import setup

setup(
    name='deep_light_field_interp',
    version='0.1',
    packages=['deeplfinterp',
              'deeplfinterp.util',
              'deeplfinterp.tools',
              'deeplfinterp.models',
              'deeplfinterp.datasets',
              'deeplfinterp.experiments',
              'deeplfinterp.ext',
              'deeplfinterp.ext.PerceptualSimilarity',
              'deeplfinterp.ext.PerceptualSimilarity.models',
              'deeplfinterp.ext.PerceptualSimilarity.util',
              'deeplfinterp.ext.PerceptualSimilarity.weights',
              'deeplfinterp.ext.PerceptualSimilarity.weights/v00',
              'deeplfinterp.ext.PerceptualSimilarity.weights/v01'
              ],
    url='',
    license='',
    author='Seán Bruton',
    author_email='sbruton@tcd.ie',
    description='Light Field Interpolation',
    install_requires=['torch',
                      'numpy',
                      'h5py',
                      'progressbar2',
                      'matplotlib',
                      'sklearn',
                      'scikit-image']
)
