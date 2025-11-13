import codecs
import os
import re

from setuptools import setup, find_packages

cur_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(cur_dir, 'README.md'), 'rb') as f:
    lines = [x.decode('utf-8') for x in f.readlines()]
    lines = ''.join([re.sub('^<.*>\n$', '', x) for x in lines])
    long_description = lines


def read(*parts):
    with codecs.open(os.path.join(cur_dir, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]",
        version_file,
        re.M,
    )
    if version_match:
        return version_match.group(1)

    raise RuntimeError("Unable to find version string.")


setup(
    name='pogema',
    author='Alexey Skrynnik',
    license='MIT',
    version=find_version("pogema", "__init__.py"),
    description='Partially Observable Grid Environment for Multiple Agents',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AIRI-Institute/pogema',
    install_requires=[
        "gymnasium==0.28.1",
        "numpy>1.23.5,<=1.26.4",
        "pydantic>=1.8.2,<=1.9.1",
    ],
    extras_require={

    },
    package_dir={'': './'},
    packages=find_packages(where='./', include='pogema*'),
    include_package_data=True,
    python_requires='>=3.7',
)
