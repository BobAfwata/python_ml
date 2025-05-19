$start = Get-Date

python train.py

$end = Get-Date
$duration = $end - $start

Write-Host "Execution time: $($duration.TotalSeconds) seconds"
