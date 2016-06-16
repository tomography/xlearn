#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='convnet',
    author='Xiaogang Yang, Francesco De Carlo',
    packages=find_packages(),
    version=open('VERSION').read().strip(),
    description = 'Convolutional Neural Network for X-ray Science.',
    license='BSD-3',
    platforms='Any',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: BSD-3',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
