from setuptools import setup, find_packages

setup(
    name='any_precision_llm',
    version='0.0.0',
    packages=find_packages(),
    package_data={'any_precision': ['analyzer/architectures/*.yaml']},
    entry_points={'vllm.general_plugins': [
        'anyprecision=any_precision.integrations.vllm:register',
    ]},
)
