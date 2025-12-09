from setuptools import setup, find_packages

import versioneer

setup(
    name="chord_pipeline",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    python_requires=">=3.7",
    author="CHORD Collaboration",
    author_email="seth.siegel@mcgill.ca",
    description="CHORD Pipeline",
    url="http://github.com/chord-observatory/chord_pipeline/",
    license="MIT",
)
