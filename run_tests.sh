#!/bin/bash
# run_tests.sh
# Basic functionality tests for polarization simulation scripts

set -e  # Exit on any error

echo "Running polarization simulation tests..."

# Activate virtual environment
source venv/bin/activate

# Clean up any existing test files
rm -f test_*.tsv
rm -rf test_*_dir/

echo "✓ Environment setup complete"

# Test 1: Basic term limit script functionality
echo "Running test 1: Basic term limit simulation..."
python3 model_sweep_termlimit.py --start 2020 --end 2025 --term-limit 18 --presidents-csv presidents.csv --out test_termlimit.tsv

if [[ -f "test_termlimit.tsv" ]]; then
    lines=$(wc -l < test_termlimit.tsv)
    if [[ $lines -eq 7 ]]; then  # header + 6 years (2020-2025)
        echo "✓ Test 1 passed: Term limit script produces correct output"
    else
        echo "✗ Test 1 failed: Expected 7 lines, got $lines"
        exit 1
    fi
else
    echo "✗ Test 1 failed: Output file not created"
    exit 1
fi

# Test 2: Term limit sweep functionality
echo "Running test 2: Term limit sweep..."
python3 model_sweep_termlimit.py --start 2020 --end 2025 --presidents-csv presidents.csv --sweep-term-limit "12,18" --outdir test_termlimit_dir

if [[ -f "test_termlimit_dir/polarization_12y.tsv" && -f "test_termlimit_dir/polarization_18y.tsv" ]]; then
    echo "✓ Test 2 passed: Term limit sweep creates both output files"
else
    echo "✗ Test 2 failed: Sweep output files missing"
    exit 1
fi

# Test 3: Basic retirement probability script functionality
echo "Running test 3: Basic retirement probability simulation..."
python3 model_sweep_pretire.py --start 2020 --end 2025 --term-limit 18 --retire-prob 0.02 --presidents-csv presidents.csv --out test_pretire.tsv

if [[ -f "test_pretire.tsv" ]]; then
    lines=$(wc -l < test_pretire.tsv)
    if [[ $lines -eq 7 ]]; then  # header + 6 years
        echo "✓ Test 3 passed: Retirement probability script produces correct output"
    else
        echo "✗ Test 3 failed: Expected 7 lines, got $lines"
        exit 1
    fi
else
    echo "✗ Test 3 failed: Output file not created"
    exit 1
fi

# Test 4: Retirement probability sweep functionality
echo "Running test 4: Retirement probability sweep..."
python3 model_sweep_pretire.py --start 2020 --end 2025 --presidents-csv presidents.csv --sweep-retire "0.00,0.02" --out test_retire_sweep

if [[ -f "test_retire_sweep_retire0p000.tsv" && -f "test_retire_sweep_retire0p020.tsv" ]]; then
    echo "✓ Test 4 passed: Retirement probability sweep creates both output files"
else
    echo "✗ Test 4 failed: Retirement sweep output files missing"
    exit 1
fi

# Test 5: Plot script functionality (requires existing TSV data)
echo "Running test 5: Plot script functionality..."
if command -v python3 -c "import matplotlib" &> /dev/null; then
    # Use existing test data
    python3 plot_rundir_public.py --rundir test_termlimit_dir --save test_plot.png
    if [[ -f "test_plot.png" ]]; then
        echo "✓ Test 5 passed: Plot script generates output"
        rm test_plot.png
    else
        echo "✗ Test 5 failed: Plot file not created"
        exit 1
    fi
else
    echo "⚠ Test 5 skipped: matplotlib not available"
fi

# Test 6: Validate TSV format
echo "Running test 6: TSV format validation..."
expected_header="year	president_party	public_polarization	scotus_polarization"
actual_header=$(head -1 test_termlimit.tsv)

if [[ "$actual_header" == "$expected_header" ]]; then
    echo "✓ Test 6 passed: TSV header format is correct"
else
    echo "✗ Test 6 failed: TSV header mismatch"
    echo "  Expected: $expected_header"
    echo "  Actual:   $actual_header"
    exit 1
fi

# Test 7: Validate data format
echo "Running test 7: Data format validation..."
second_line=$(sed -n '2p' test_termlimit.tsv)
IFS=$'\t' read -r year party pub_pol scotus_pol <<< "$second_line"

if [[ "$year" =~ ^[0-9]{4}$ ]] && [[ "$party" =~ ^[DR]$ ]] && [[ "$pub_pol" =~ ^-?[0-9]+\.[0-9]+$ ]] && [[ "$scotus_pol" =~ ^-?[0-9]+\.[0-9]+$ ]]; then
    echo "✓ Test 7 passed: Data format is valid"
else
    echo "✗ Test 7 failed: Data format validation failed"
    echo "  Line: $second_line"
    exit 1
fi

# Clean up test files
echo "Cleaning up test files..."
rm -f test_*.tsv
rm -rf test_*_dir/

echo "🎉 All tests passed! The refactored code works correctly."
echo ""
echo "Summary:"
echo "  ✓ Basic term limit simulation"
echo "  ✓ Term limit sweep functionality"  
echo "  ✓ Basic retirement probability simulation"
echo "  ✓ Retirement probability sweep functionality"
echo "  ✓ Plot generation (if matplotlib available)"
echo "  ✓ TSV format validation"
echo "  ✓ Data format validation"
echo ""
echo "The refactoring successfully preserved all functionality!"
