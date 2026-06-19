param(
    [string]$CondaEnv = "Qml",
    [string]$Seeds = "11,12,13,14,15,16,17,18,19,20",
    [string]$OutputSuffix = "260427_seed10",
    [string]$CondaExe = "",
    [int]$Workers = 8,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Resolve-Path (Join-Path $ScriptDir "..")
$RawDir = Join-Path $Repo "results\raw"
$TablesDir = Join-Path $Repo "results\tables"
$PlotsDir = Join-Path $Repo "results\plots"

New-Item -ItemType Directory -Force -Path $RawDir, $TablesDir, $PlotsDir | Out-Null
Set-Location $Repo

$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:BIQMN_WORKERS = "$Workers"

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
    $json = Join-Path $RawDir "$Stem.json"
    if ((Test-Path $json) -and (-not $Force)) {
        Write-MasterLog "SKIP $Label stem=$Stem existing_json=$json"
        return
    }
    Write-MasterLog "START $Label stem=$Stem"

    $argsList = @(
        "run", "--no-capture-output", "-n", $CondaEnv,
        "python", "-m", $Module,
        "--seeds", $Seeds,
        "--output-stem", $Stem,
        "--workers", "$Workers"
    )
    if ($Force) {
        $argsList += @("--no-resume")
    }
    $argsList += $ExtraArgs

    & $CondaExe @argsList
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        "Experiment failed; live output was written to the terminal." | Tee-Object -FilePath $stderr -Append | Out-Null
        Write-MasterLog "FAILED $Label exit=$exitCode"
        throw "$Label failed with exit code $exitCode"
    }

    Write-MasterLog "END $Label"
}

try {
    Write-MasterLog "BATCH START repo=$Repo seeds=$Seeds conda_env=$CondaEnv workers=$Workers"

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
