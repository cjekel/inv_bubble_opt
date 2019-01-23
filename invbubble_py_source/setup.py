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
    description='wip',
    long_description='wip: not ready for public release',
    platforms=['any']
)