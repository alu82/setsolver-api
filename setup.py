from setuptools import setup

setup(
    name='setsolverapi',
    packages=['setsolverapi'],
    include_package_data=True,
    install_requires=[
        'flask','torch','torchvision','opencv-python'
    ],
)