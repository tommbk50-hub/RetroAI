def pytest_addoption(parser):
    parser.addoption(
        "--cbyd-trees",
        action="store",
        default="",
        help="Run opt-in CbyD route-tree benchmarks using this route_tree.jsonl",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "benchmark: opt-in CbyD route benchmark (requires --cbyd-trees PATH)",
    )
