from setuptools import setup

setup(
    name='SFstarred',
    version='0.0.1',
    description="SFstarred",
    url='?',
    author='?',
    author_email='?',  # Optional: add if you want to display a contact
    license='CC0 1.0 Universal (Public Domain Dedication)',
    packages=['SFstarred'],
    install_requires=["starred-astro","pandas","pyregion","photutils"],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10',
    ],
)

