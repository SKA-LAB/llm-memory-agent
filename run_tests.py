import os
import sys
import coverage
import unittest
from tests.test_config import configure_test_logging

def run_tests_with_coverage():
    """Run all tests with coverage and generate a report."""
    # Configure logging to show debug messages
    configure_test_logging()
    
    # Initialize coverage.py
    cov = coverage.Coverage()
    
    # Start measuring code coverage
    cov.start()
    
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Stop measuring code coverage
    cov.stop()
    
    # Save coverage results
    cov.save()
    
    print("\n\n" + "="*80)
    print("COVERAGE REPORT")
    print("="*80)
    
    # Report coverage statistics
    cov.report(show_missing=True)  # Show missing lines in the report
    
    # Generate HTML report
    cov_dir = os.path.join(os.path.dirname(__file__), 'htmlcov')
    cov.html_report(directory=cov_dir)
    
    print(f"\nDetailed HTML coverage report generated at: {cov_dir}/index.html")
    
    # Return test result to use as exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    exit_code = run_tests_with_coverage()
    exit(exit_code)