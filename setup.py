from pathlib import Path
from setuptools import find_packages, setup

# Load version number
__version__ = ''
version_file = Path(__file__).parent.absolute() / 'trade' / '_version.py'

with open(version_file) as fd:
    exec(fd.read())

# Load README
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()
with open('requirements.txt', encoding='utf-8') as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith('#')
    ]

setup(
    name='trade',
    version=__version__,
    description='funnel-like model to screen the drug for nano-particles',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/',
    download_url=f'',
    license='MIT',
    packages=find_packages(),
    package_data={'synthemol': ['py.typed', 'resources/**/*']},
    entry_points={
        'console_scripts': [
            'trade=trade.screen.screening:generate_command_line'
        ]
    },
    install_requires=requirements,
    python_requires='>=3.10',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Typing :: Typed'
    ],
    keywords=[
        'machine learning',
        'drug design',
        'Au nano-particle'
    ]
)
