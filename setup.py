#!/usr/bin/env python3

from setuptools import setup
from setuptools import find_packages

setup(name='ganrecs',
      version='1.0',
      description='Experiments for GAN-Based Recommendations',
      author=[
          "Austin Graham",
          "Carlos Sanchez"
      ],
      author_email=[
          "austin.graham@nextthought.com",
          "carlos.sanchez@nextthought.com"
      ],
      url='https://github.com/austinpgraham/Recommendations-GAN',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      entry_points = {
          'console_scripts': [
              'ganrecs_mnist_test=ganrecs.scripts.ganrecs_mnist:main'
          ]
      },
      install_requires=[
          'tensorflow',
          'numpy',
          'matplotlib'
      ]
)