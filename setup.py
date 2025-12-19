from setuptools import find_packages, setup

package_name = 'arm'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='karthik',
    maintainer_email='karthik@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'control_node = arm.control:main',
            'task1c = arm.task1c:main',
            'task1b = arm.task1b:main',
            'ebot_nav_task1A = arm.ebot_nav_task1A:main',
            'task2A = arm.ebot_nav_task2a:main',
            'shape = arm.shape_detector_task2a:main',
            'task3 = arm.task3:main',
            'task3a_tf_publisher = arm.task3a_tf_publisher:main',
            'task3b_ur5_servo_pick_place = arm.task3b_ur5_servo_pick_place:main',
            'man = arm.task3b_manipulation:main',
            'per = arm.task3b_perception:main',
            'task4a_m = arm.task4a_manipulation:main',
            'task4a_p = arm.task4a_perception:main',

            
        ],
    },
)
