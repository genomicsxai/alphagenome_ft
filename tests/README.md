# AlphaGenome-FT Tests

This directory contains the test suite for the `alphagenome-ft` package.

## Test Structure

```
tests/
├── __init__.py                      # Package marker
├── conftest.py                      # Pytest fixtures and shared setup
├── test_custom_heads.py             # Tests for custom head functionality
├── test_model_predictions.py        # Tests for model prediction consistency
├── test_encoder_only_mode.py        # Tests for encoder-only mode (short sequences)
├── test_parameter_management.py     # Tests for parameter freezing
└── README.md                        # This file
```

## Running Tests

### Install Test Dependencies

```bash
pip install -e ".[test]"
# Or for development (includes all dev tools):
pip install -e ".[dev]"
```

### Run All Tests

```bash
# From package root directory
pytest

# With verbose output
pytest -v

# With coverage report
pytest --cov=alphagenome_ft --cov-report=html
```

### Run Specific Tests

```bash
# Run single test file
pytest tests/test_custom_heads.py

# Run specific test class
pytest tests/test_custom_heads.py::TestHeadRegistry

# Run specific test
pytest tests/test_custom_heads.py::TestHeadRegistry::test_register_custom_head

# Run tests matching pattern
pytest -k "parameter"
```

### Run with Markers

```bash
# Run only unit tests (if marked)
pytest -m unit

# Skip slow tests
pytest -m "not slow"
```

## Test Categories

### 1. Custom Head Tests (`test_custom_heads.py`)

Tests for custom head registration and configuration:
- Head registration and retrieval
- Configuration management
- Registry operations
- CustomHead base class functionality

### 2. Model Prediction Tests (`test_model_predictions.py`)

Tests for model prediction consistency:
- Wrapped model produces identical predictions to base model
- Custom head outputs are correctly structured
- Parameter preservation in custom-only models

### 3. Encoder-Only Mode Tests (`test_encoder_only_mode.py`)

Tests for encoder-only mode (short sequence support):
- Model creation with `use_encoder_output=True`
- Short sequence inference (< 1000 bp)
- Encoder embeddings availability
- Parameter freezing in encoder-only mode

### 4. Parameter Management Tests (`test_parameter_management.py`)

Tests for parameter inspection and freezing:
- Parameter counting and path listing
- Backbone vs head parameter identification
- Freezing and unfreezing functionality
- Parameter value preservation

## Fixtures

Key fixtures defined in `conftest.py`:

- **`device`**: Compute device (CPU for testing)
- **`base_model`**: Pretrained AlphaGenome model (session-scoped)
- **`test_interval`**: Genomic interval for testing
- **`test_sequence`**: One-hot encoded DNA sequence
- **`registered_mpra_head`**: Registered test MPRA head
- **`wrapped_model_with_head`**: Wrapped model with custom head
- **`custom_only_model`**: Model with only custom heads

## Test Data

Tests use a genomic interval from chromosome 22 (hg38):
- Chromosome: chr22
- Start: 35,677,410
- End: 36,725,986
- Size: ~1 Mbp

This provides a realistic test case without requiring large downloads.

## Writing New Tests

### Example Test Structure

```python
import pytest

class TestMyFeature:
    """Test suite for my feature."""
    
    def test_something(self, base_model, test_sequence):
        """Test that something works correctly."""
        # Arrange
        input_data = prepare_input(test_sequence)
        
        # Act
        result = my_function(base_model, input_data)
        
        # Assert
        assert result is not None
        assert result.shape == expected_shape
```

### Using Fixtures

```python
def test_with_fixtures(
    base_model,           # Session-scoped: reused across tests
    wrapped_model_with_head,  # Function-scoped: fresh for each test
):
    """Fixtures are automatically injected by pytest."""
    # Use fixtures directly
    predictions = wrapped_model_with_head.predict(...)
```

### Adding Markers

```python
@pytest.mark.slow
def test_expensive_operation():
    """Mark slow tests so they can be skipped if needed."""
    pass

@pytest.mark.integration
def test_full_pipeline():
    """Mark integration tests separately from unit tests."""
    pass
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e ".[test]"
      - name: Run tests
        run: pytest --cov=alphagenome_ft --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Performance Considerations

- **Session-scoped fixtures**: `base_model` is loaded once per test session to avoid repeated downloads
- **Function-scoped fixtures**: Models with custom heads are recreated for each test to ensure isolation
- **Test data**: Uses a single genomic region to minimize data loading

## Troubleshooting

### Tests Fail with "Module not found"

Make sure the package is installed in editable mode:
```bash
pip install -e .
```

### Tests Are Very Slow

The first test run downloads the AlphaGenome model (~few GB). Subsequent runs use cached data.

To skip slow tests:
```bash
pytest -m "not slow"
```

### JAX/CUDA Issues

Tests use CPU by default. If you encounter GPU-related errors:
```bash
export JAX_PLATFORM_NAME=cpu
pytest
```

## Coverage

Generate coverage report:
```bash
pytest --cov=alphagenome_ft --cov-report=html
# Open htmlcov/index.html in browser
```

Target coverage: >80% for core functionality

## Contributing

When adding new features:
1. Write tests first (TDD)
2. Ensure all tests pass: `pytest`
3. Check coverage: `pytest --cov`
4. Add docstrings to test functions
5. Update this README if adding new test categories


