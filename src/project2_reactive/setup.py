from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'project2_reactive'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.py'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Riley',
    maintainer_email='student@ou.edu',
    description=(
        'CS 4023 Project 2: Reactive Robotics using ROS 2 and TurtleBot 4'
    ),
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'reactive_controller = '
            'project2_reactive.reactive_controller:main',
        ],
    },
)
