from setuptools import setup

setup(
    name='gym-submit',
    author="Max Lapan",
    author_email="max.lapan@gmail.com",
    license='GPL-v3',
    version='0.1',
    description="Tool to submit solution to OpenAI Gym",
    instal_requires=['gym'],
    scripts=["gym-submit.py"],
)
