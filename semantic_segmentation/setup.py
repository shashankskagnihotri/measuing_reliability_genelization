from setuptools import setup, find_packages


setup(
    name='semsegbench',
    version='0.1.0',
    # author='Your Name',
    # author_email='your.email@example.com',
    description='A package for benchmarking the robustness of semantic segmentation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/shashankskagnihotri/benchmarking_robustness/',
    packages=find_packages(),
    install_requires=[
        
    ],
    license='MIT',
)