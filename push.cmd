@echo off
rem 用仓库内自带的部署密钥推送（Windows）
rem 用法： push.cmd            或    push.cmd origin main
rem
rem 为什么需要它：本机密钥在 .deploy\ssh\ 而不是 ~/.ssh，
rem 且 Git for Windows 自带 sh 在受限环境里可能起不来，所以显式指定 GIT_SSH。

setlocal
set "ROOT=%~dp0"
set "GIT_SSH=%ROOT%.deploy\ssh\ssh_wrap.cmd"
set "GIT_SSH_VARIANT=ssh"

if not exist "%GIT_SSH%" (
    echo [x] 找不到 .deploy\ssh\ssh_wrap.cmd
    echo     这台机器上没有本仓库的部署密钥。
    echo     可以改用 HTTPS + PAT，或生成新密钥并把公钥加到 GitHub。
    exit /b 1
)

set "REMOTE=%~1"
if "%REMOTE%"=="" set "REMOTE=origin"
set "BRANCH=%~2"
if "%BRANCH%"=="" set "BRANCH=main"

echo Pushing %BRANCH% -> %REMOTE% ...
git push %REMOTE% %BRANCH%
exit /b %ERRORLEVEL%
