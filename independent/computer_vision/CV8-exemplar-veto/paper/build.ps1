# Rebuild tables, figures and the PDF with pdfLaTeX and BibTeX (any TeX Live or TinyTeX on PATH).
# The qualitative figure needs the GPU and is rebuilt only with -Qualitative.
# Do not build with XeTeX or Tectonic: the CVPR style then falls back from Times to Latin Modern.
param([switch]$Qualitative, [switch]$PdfOnly, [string]$Python = "python")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not $PdfOnly) {
    & $Python make_tables.py
    & $Python make_figures.py
    if ($Qualitative) { & $Python make_qualitative.py }
}
cmd /c "pdflatex -interaction=nonstopmode -halt-on-error main.tex >nul 2>&1"
if ($LASTEXITCODE -ne 0) { Get-Content main.log -Tail 40; throw "pdflatex failed" }
cmd /c "bibtex main >nul 2>&1"
if ($LASTEXITCODE -ne 0) { Get-Content main.blg; throw "bibtex failed" }
cmd /c "pdflatex -interaction=nonstopmode -halt-on-error main.tex >nul 2>&1"
cmd /c "pdflatex -interaction=nonstopmode -halt-on-error main.tex >nul 2>&1"
cmd /c "pdflatex -interaction=nonstopmode -halt-on-error main.tex >nul 2>&1"
# One more pass: from a clean build the review-mode line numbers on the reference pages need it to settle.
cmd /c "pdflatex -interaction=nonstopmode -halt-on-error main.tex >nul 2>&1"
Select-String -Path main.log -Pattern "^!|Overfull|undefined|multiply defined|Rerun to|Output written" | ForEach-Object { $_.Line }
Remove-Item main.aux, main.blg, main.brf, main.log, main.out -ErrorAction SilentlyContinue
