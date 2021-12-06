for ((i=1; i<=100; i++))
do
    python capture.py -r baselineTeam -b myTeam -l RANDOM"$i" --q >> resultsBlue.txt
    python capture.py -r myTeam -b baselineTeam -l RANDOM"$i" --q >> resultsRed.txt
done