param(
    [string]$CondaEnv = "QEC",
    [string]$Seeds = "11,12,13,14,15,16,17,18,19,20",
    [string]$OutputSuffix = "260427_seed10",
    [string]$CondaExe = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Resolve-Path (Join-Path $ScriptDir "..")
$RawDir = Join-Path $Repo "results\raw"
$TablesDir = Join-Path $Repo "results\tables"
$PlotsDir = Join-Path $Repo "results\plots"

New-Item -ItemType Directory -Force -Path $RawDir, $TablesDir, $PlotsDir | Out-Null
Set-Location $Repo

if (-not $CondaExe) {
    $condaCommand = Get-Command conda -ErrorAction SilentlyContinue
    if ($condaCommand) {
        $CondaExe = $condaCommand.Source
    }
    else {
        $candidate = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"
        if (Test-Path $candidate) {
            $CondaExe = $candidate
        }
        else {
            throw "Could not find conda. Pass -CondaExe with the full path to conda.exe."
        }
    }
}

$MasterLog = Join-Path $RawDir "c3r_$OutputSuffix.batch.stdout.log"
$MasterErr = Join-Path $RawDir "c3r_$OutputSuffix.batch.stderr.log"

function Write-MasterLog {
    param([string]$Message)
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"), $Message
    $line | Tee-Object -FilePath $MasterLog -Append
}

function Invoke-C3RExperiment {
    param(
        [string]$Label,
        [string]$Module,
        [string]$Stem,
        [string[]]$ExtraArgs = @()
    )

    $stdout = Join-Path $RawDir "$Stem.stdout.log"
    $stderr = Join-Path $RawDir "$Stem.stderr.log"
    Write-MasterLog "START $Label stem=$Stem"

    $argsList = @(
        "run", "-n", $CondaEnv,
        "python", "-m", $Module,
        "--seeds", $Seeds,
        "--output-stem", $Stem
    ) + $ExtraArgs

    & $CondaExe @argsList 1>> $stdout 2>> $stderr
    if ($LASTEXITCODE -ne 0) {
        Write-MasterLog "FAILED $Label exit=$LASTEXITCODE"
        throw "$Label failed with exit code $LASTEXITCODE"
    }

    Write-MasterLog "END $Label"
}

try {
    Write-MasterLog "BATCH START repo=$Repo seeds=$Seeds conda_env=$CondaEnv"

    Invoke-C3RExperiment `
        -Label "clean_regime_map" `
        -Module "biqmn.experiments.run_hybrid_c123_regime_map" `
        -Stem "hybrid_c123_regime_map_c3r_$OutputSuffix" `
        -ExtraArgs @("--plot-prefix", "hybrid_c123_regime_map_c3r_$OutputSuffix")

    Invoke-C3RExperiment `
        -Label "partial_syndrome" `
        -Module "biqmn.experiments.run_partial_syndrome_baseline" `
        -Stem "partial_syndrome_c3r_$OutputSuffix" `
        -ExtraArgs @("--plot-prefix", "partial_syndrome_c3r_$OutputSuffix")

    Invoke-C3RExperiment `
        -Label "noisy_syndrome" `
        -Module "biqmn.experiments.run_noisy_syndrome_baseline" `
        -Stem "noisy_syndrome_c3r_$OutputSuffix" `
        -ExtraArgs @("--plot-prefix", "noisy_syndrome_c3r_$OutputSuffix")

    Invoke-C3RExperiment `
        -Label "partial_noisy_syndrome" `
        -Module "biqmn.experiments.run_partial_noisy_syndrome_regime_map" `
        -Stem "partial_noisy_syndrome_c3r_$OutputSuffix" `
        -ExtraArgs @("--plot-prefix", "partial_noisy_syndrome_c3r_$OutputSuffix")

    Invoke-C3RExperiment `
        -Label "ambiguity_measurement_syndrome" `
        -Module "biqmn.experiments.run_ambiguity_measurement_syndrome_regime_map" `
        -Stem "ambiguity_measurement_c3r_$OutputSuffix" `
        -ExtraArgs @("--plot-prefix", "ambiguity_measurement_c3r_$OutputSuffix")

    Write-MasterLog "BATCH END success"
}
catch {
    $_ | Out-String | Tee-Object -FilePath $MasterErr -Append
    Write-MasterLog "BATCH END failed"
    exit 1
}
