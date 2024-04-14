from setuptools import setup, Extension

setup(name='mykmeanssp',
      version='1.1',
      author="Ahmad, Christian",
      description='kmeans algorithm for sp class',
      ext_modules=[Extension('mykmeanssp',['kmeansmodule.c'],), ])
