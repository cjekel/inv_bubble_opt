from distutils.core import setup
setup(
    name='invbubble',
    version=open('invbubble/VERSION').read().strip(),
    author='Charles Jekel',
    author_email='cjekel@gmail.com',
    packages=['invbubble'],
    package_data={'invbubble': ['VERSION']},
    url='https://jekel.me',
    license='MIT License',
    description='Tools for identifying material parameters from bulge inflation test data.',  # noqa E501
    long_description='Tools for identifying material parameters from bulge inflation test data.',  # noqa E501
    platforms=['any'],
    install_requires=[
        "numpy >= 1.14.0",
        "scipy >= 1.2.0",
        "pyDOE >= 0.3.8",
        "pandas",
        "joblib",
    ],
)