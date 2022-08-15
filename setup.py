# coding: utf-8
from __future__ import print_function, unicode_literals
import sys
import codecs
from setuptools import setup, find_packages
from nano_indent import __version__, __author__, __email__


with open('requirements.txt') as f:
    requirements = [l for l in f.read().splitlines() if l]


def long_description():
    with codecs.open('README.md', 'rb') as readme:
        if not sys.version_info < (3, 0, 0):
            return readme.read().decode('utf-8')


setup(
    name='nano_indent',
    version=__version__,
    packages=find_packages(),

    author=__author__,
    author_email=__email__,
    keywords=['GA', 'Nano_Indent','analysis'],
    description='Nano Indent using GA',
    long_description=long_description(),
    url='https://github.com/laumiulun/Nano-Neo-Public.git',
    download_url='https://github.com/laumiulun/Nano-Neo-Public/tarball/master',
    include_package_data=True,
    zip_safe=False,
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'nano_neo = nano_indent.nano_indent:main',
        ]
    },
    license='GPLv3',
)
