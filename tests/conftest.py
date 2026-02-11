import os
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--zoo_user", action="store", default="user", help="Zooniverse username"
    )
    parser.addoption(
        "--zoo_pass", action="store", default="password", help="Zooniverse password"
    )


@pytest.fixture
def zoo_user(request):
    return request.config.getoption("--zoo_user")


@pytest.fixture
def zoo_pass(request):
    return request.config.getoption("--zoo_pass")


@pytest.fixture
def needs_wandb():
    if not os.environ.get("WANDB_API_KEY"):
        pytest.skip("No WANDB_API_KEY")
