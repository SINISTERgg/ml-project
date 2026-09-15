"""Root conftest — deregisters the broken web3 pytest_ethereum plugin."""
import sys

# The web3 package ships a pytest11 entry-point (pytest_ethereum) that imports
# a name that no longer exists in the installed eth_typing version, crashing
# pytest at startup.  We deregister it here before it can load.
collect_ignore_glob = []

def pytest_configure(config):
    try:
        config.pluginmanager.unregister(name="pytest_ethereum")
    except Exception:
        pass
