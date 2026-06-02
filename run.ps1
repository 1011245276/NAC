# NAC Quick Run (Windows PowerShell)
# Usage: .\run.ps1 -Method nac -Dataset cifar10 -Eps 4
param(
    [string]$Method = "nac",
    [string]$Dataset = "cifar10",
    [int]$Eps = 4
)
Write-Host "NAC: method=$Method dataset=$Dataset eps=$Eps/255"
python nac_fair_experiment.py `
    --batch_size 32 `
    --root ./data `
    --test_set $Dataset `
    --test_attack_type pgd `
    --test_eps $Eps `
    --test_numsteps 10 `
    --test_stepsize 1 `
    --ttc_eps 4 `
    --ttc_numsteps 2 `
    --ttc_stepsize 1 `
    --tau_thres 0.2 `
    --beta 2 `
    --seed 0 `
    --counterattack $Method `
    --nac_momentum 0.9 `
    --outdir ./results
