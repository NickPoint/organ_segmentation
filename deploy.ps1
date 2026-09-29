<#
Push code to the cluster, submit a training job, check status, and pull results back.
PowerShell analog of deploy.sh -- same behavior, using Windows' built-in ssh.exe/scp.exe
(OpenSSH client, present by default on Windows 10/11).

Requires passwordless (key-based) SSH to run non-interactively; otherwise each ssh/scp
call will just prompt for your password/2FA, same as running them by hand.

Usage:
  .\deploy.ps1 setup                                              # one-time: create venv + install requirements.txt
  .\deploy.ps1 push                                               # copy the train_*.py scripts / train.job / requirements.txt over
  .\deploy.ps1 submit -TrainScript train_25d.py -Architecture unetplusplus   # push, then submit train.job
  .\deploy.ps1 status                                             # show your jobs in the queue
  .\deploy.ps1 pull -JobId 123                                    # copy back results/123* and its logs
#>

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("setup", "push", "submit", "status", "pull")]
    [string]$Command,

    [string]$JobId,
    [string]$TrainScript,
    [string]$Architecture
)

$RemoteHost = "voievuds@rocket.hpc.ut.ee"
$RemoteDir = "dimas_challenge"

function Invoke-Push {
    $scripts = (Get-ChildItem train_*.py).Name
    Write-Host "Copying $($scripts -join ', '), train.job, requirements.txt to ${RemoteHost}:${RemoteDir}..."
    & scp @scripts train.job requirements.txt "${RemoteHost}:${RemoteDir}/"
    if ($LASTEXITCODE -ne 0) { throw "scp failed" }
    & ssh $RemoteHost "mkdir -p $RemoteDir/logs $RemoteDir/results"
}

switch ($Command) {
    "setup" {
        Write-Host "Copying requirements.txt and creating a venv on $RemoteHost..."
        & scp requirements.txt "${RemoteHost}:${RemoteDir}/"
        if ($LASTEXITCODE -ne 0) { throw "scp failed" }

        & ssh $RemoteHost "cd $RemoteDir && module load python && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
        if ($LASTEXITCODE -ne 0) { throw "remote setup failed" }

        Write-Host ""
        Write-Host "Venv created. torch is NOT installed by this step on purpose -- install it"
        Write-Host "yourself, matched to the falcon nodes' Tesla V100 GPUs (compute capability 7.0,"
        Write-Host "which newer cu13x builds no longer ship kernels for -- use cu126 instead):"
        Write-Host "  ssh $RemoteHost"
        Write-Host "  cd $RemoteDir; source .venv/bin/activate; pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126"
    }

    "push" {
        Invoke-Push
    }

    "submit" {
        if ([string]::IsNullOrEmpty($TrainScript) -or [string]::IsNullOrEmpty($Architecture)) {
            Write-Host "Usage: .\deploy.ps1 submit -TrainScript <script> -Architecture <architecture>"
            Write-Host "  e.g. .\deploy.ps1 submit -TrainScript train_25d.py -Architecture unetplusplus"
            exit 1
        }
        if (-not (Test-Path $TrainScript)) {
            Write-Host "$TrainScript not found here"
            exit 1
        }

        Invoke-Push
        # the job name becomes part of the log file names and of the results folder
        $name = "${Architecture}_$([System.IO.Path]::GetFileNameWithoutExtension($TrainScript))"
        Write-Host "Submitting $name ..."
        & ssh $RemoteHost "cd $RemoteDir && sbatch --job-name=$name --export=ALL,SCRIPT=$TrainScript,ARCHITECTURE=$Architecture train.job"
    }

    "status" {
        & ssh $RemoteHost 'squeue -u $USER'
    }

    "pull" {
        if ([string]::IsNullOrEmpty($JobId)) {
            Write-Host "Usage: .\deploy.ps1 pull -JobId <job_id>"
            exit 1
        }

        New-Item -ItemType Directory -Force -Path "results" | Out-Null
        New-Item -ItemType Directory -Force -Path "logs" | Out-Null

        Write-Host "Pulling results/$JobId* ..."
        $remoteRuns = & ssh $RemoteHost "ls -d $RemoteDir/results/${JobId}* 2>/dev/null"
        foreach ($line in $remoteRuns) {
            $remoteRun = $line.Trim()
            if ($remoteRun) {
                & scp -r "${RemoteHost}:$remoteRun" "results\"
            }
        }

        Write-Host "Pulling logs for job $JobId ..."
        $remoteLogs = & ssh $RemoteHost "ls $RemoteDir/logs/*_${JobId}[_.]* 2>/dev/null"
        foreach ($line in $remoteLogs) {
            $remoteFile = $line.Trim()
            if ($remoteFile) {
                & scp "${RemoteHost}:$remoteFile" "logs\"
            }
        }
    }
}
