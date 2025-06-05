from setuptools import setup

setup(
    name="f110_gym",
    version="0.2.1",
    author="Hongrui Zheng",
    author_email="billyzheng.bz@gmail.com",
    url="https://f1tenth.org",
    package_dir={"": "gym"},
    install_requires=[
        "gym==0.23.1",
        "numpy==1.26.4",
        "Pillow==11.1.0",
        "scipy==1.15.1",
        "numba==0.61.0",
        "pyyaml==6.0.2",
        "pyglet==1.4.11",
        "pyopengl==3.1.9",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "tqdm==4.66.5",
        "pandas==2.2.3",
    ],
)
