param(
    [string]$CondaEnv = "Qml",
    [string]$Stems = "",
    [string]$CondaExe = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = Resolve-Path (Join-Path $ScriptDir "..")
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

$argsList = @("run", "--no-capture-output", "-n", $CondaEnv, "python", "-m", "biqmn.experiments.rebuild_c3r_results")
if ($Stems) {
    $argsList += @("--stems", $Stems)
}

& $CondaExe @argsList
if ($LASTEXITCODE -ne 0) {
    throw "C3R table rebuild failed with exit code $LASTEXITCODE"
}
