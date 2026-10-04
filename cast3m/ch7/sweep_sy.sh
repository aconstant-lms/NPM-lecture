#!/bin/bash
# sweep_sy.sh -- companion exercise C7.3: the cost J(sY) of Exercise 7.9 along
# one parameter, with E and H at their true values, from the shell alone:
# sed fills the template, Cast3M runs in its own directory, awk computes J.
# Usage: ./sweep_sy.sh            (CASTEM=castem26 by default)
#        CASTEM="python3 $PWD/tools/castem_stub.py" ./sweep_sy.sh   (test)
set -eu
CASTEM=${CASTEM:-castem26}
E=200000.; H=10000.; UREF=0.001            # u_ref = sY l / E (true values)
HERE=$(cd "$(dirname "$0")" && pwd)

printf "%8s %12s\n" "sY" "J"
for SY in 120 140 160 180 190 195 200 205 210 220 240 260 280; do
  d="$HERE/runs/sweep_sY$SY"
  mkdir -p "$d"
  sed -e "s/__E__/$E/" -e "s/__SY__/$SY./" -e "s/__H__/$H/" \
      "$HERE/truss.template" > "$d/calcul.dgibi"
  ( cd "$d" && $CASTEM calcul.dgibi > sortie.log 2>&1 )
  if grep -q ERREUR "$d/sortie.log" || [ ! -f "$d/truss_u.csv" ]; then
    echo "FAILED: sY = $SY (see $d/sortie.log)"; continue
  fi
  # J = 1/2 sum_n |u_n - u^m_n|^2 / u_ref^2: paste the computed and measured
  # files side by side and find the columns UX, UY of each half by their
  # header (EXPORTCSV writes the columns in the order of INDEX, not ours)
  paste -d';' "$d/truss_u.csv" "$HERE/truss_measurements.csv" |
    awk -F';' -v sy="$SY" -v uref="$UREF" '
      NR == 1 { h = NF / 2
                for (i = 1; i <= h; i++) { if ($i == "UX") cx = i; if ($i == "UY") cy = i }
                for (i = h + 1; i <= NF; i++) { if ($i == "UX") mx = i; if ($i == "UY") my = i }
                next }
              { J += ($cx - $mx)^2 + ($cy - $my)^2 }
      END     { printf "%8s %12.4f\n", sy, 0.5 * J / uref^2 }'
done
# Output with the Python stand-in (tools/castem_stub.py):
#   sY = 120 160 190 200 210 240 280: J = 6320 1945 122.0 0.297 122.4 584.5 620.8
# a sharp valley at sY = 200 (J = 0.297, the noise level), steep below, and a
# plateau above about 240: the truss then hardly yields and sY becomes invisible.
