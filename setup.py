from setuptools import setup, find_packages

setup(
    name="wune",
    version="1.2.1",
    description="80s-90s mini-hifi-inspired LED audio spectrum visualizer",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Yoshihide Sonoda",
    url="https://github.com/yshdsnd/wune",
    license="BSD-2-Clause",
    packages=find_packages(),
    package_data={
        "wune": [
            "assets/*",
            "locales/*.json",
        ],
    },
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pygame>=2.6.1",
        "SoundCard>=0.4.6",
    ],
    entry_points={
        "console_scripts": [
            "wune = wune.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: X11 Applications",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX :: BSD :: FreeBSD",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
)
