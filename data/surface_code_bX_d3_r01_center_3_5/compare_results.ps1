param(
    [string]$FileActual = ".\obs_flips_actual_new.01",
    [string]$FilePredicted
)

# Exit with an error if the predicted file is not specified
if (-not $FilePredicted) {
    Write-Host "ERROR: You must specify a prediction file to compare." -ForegroundColor Red
    Write-Host "Example: ./compare_results.ps1 -FilePredicted obs_flips_predicted_by_pymatching.01"
    exit
}

Write-Host "Comparing: $($FileActual) vs $($FilePredicted)"

# Read the contents of both files
$actual_flips = Get-Content $FileActual
$predicted_flips = Get-Content $FilePredicted

# --- Robustness Check ---
# Check if files were read correctly and are not empty
if ($actual_flips.Length -eq 0) {
    Write-Host "ERROR: The actual flips file '$($FileActual)' is empty or could not be read." -ForegroundColor Red
    exit
}
if ($predicted_flips.Length -eq 0) {
    Write-Host "ERROR: The predicted flips file '$($FilePredicted)' is empty or could not be read." -ForegroundColor Red
    exit
}
if ($actual_flips.Length -ne $predicted_flips.Length) {
    Write-Host "WARNING: Files have a different number of lines. Comparison will be inaccurate." -ForegroundColor Yellow
}
# --- End of Check ---


# Initialize a counter for mismatches
$error_count = 0

# Determine the number of lines to compare (use the smaller of the two)
$comparison_limit = [System.Math]::Min($actual_flips.Length, $predicted_flips.Length)

# Compare each line
for ($i = 0; $i -lt $comparison_limit; $i++) {
    if ($actual_flips[$i] -ne $predicted_flips[$i]) {
        $error_count++
    }
}

# --- Display Results ---
$total_shots = $actual_flips.Length
$logical_error_rate = $error_count / $total_shots

Write-Host "----------------------------------------"
Write-Host "Decoder: $($FilePredicted)"
Write-Host "Total Shots: $($total_shots)"
Write-Host "Total Mismatches (Logical Errors): $($error_count)"
Write-Host "Logical Error Rate: $($logical_error_rate)"
Write-Host "----------------------------------------"