"""Native tests must work on a clean checkout, independent of test ordering."""
import pytest
from flyholdem.neural.kernel.build import LIBRARY, build, verify_build


@pytest.fixture(scope='session', autouse=True)
def verified_native_kernel():
    if not LIBRARY.is_file() or not LIBRARY.with_suffix(LIBRARY.suffix + '.json').is_file():
        build()
    else:
        # Reuse a verified binary; do not replace one used by an active run.
        verify_build()
