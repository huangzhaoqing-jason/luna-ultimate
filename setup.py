from setuptools import setup, find_packages

setup(
    name="luna-brain",
    version="0.2.0",
    description="Luna Brain: From-AGI-to-ASI × AIXI × 246-area functional open-source agent",
    author="Huang Zhaoqing",
    license="Apache-2.0",
    packages=find_packages(include=["brain", "brain.*", "scripts", "scripts.*"]),
    py_modules=["config_brain"],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "safetensors>=0.4.0",
    ],
    extras_require={
        "data": ["datasets>=2.18.0", "huggingface_hub>=0.20.0", "pyyaml>=6.0"],
    },
    entry_points={
        "console_scripts": [
            "luna-brain=scripts.run_brain:main",
        ],
    },
)
