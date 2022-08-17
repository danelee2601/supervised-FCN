"""
reference: https://bit.ly/3QP48AN
"""
from setuptools import setup, find_packages
from pipreqs.pipreqs import parse_requirements

install_reqs = parse_requirements('requirements.txt')
install_reqs = [f'{item["name"]}=={item["version"]}' for item in install_reqs]

setup(
    name="supervised-fcn",  # pypi 에 등록할 라이브러리 이름
    version="0.4.0",  # pypi 에 등록할 version (수정할 때마다 version up을 해줘야 함)
    description="it provides a pretrained FCN (Fully Convolutional Network).",
    author="Daesoo Lee",
    author_email="daesoolee2601@gmail.com",
    url="https://github.com/danelee2601/supervised-FCN",
    python_requires=">= 3.8",
    packages=find_packages(),
    install_requires=install_reqs,
    zip_safe=False,
    # 중요한 부분
    # entry_points={
    #     "console_scripts": [
    #         "hey = insutance.main:main"
    #     ]
    # },
    package_data={},
    include_package_data=True
)
