headerRead="0"
file="${1%.csv}"
lno="0"
perFile=$2
fno="1"
outname="$file-$fno.csv"
while IFS= read -r line 
do
	if [ $headerRead -eq 0 ] 
	then
		headerRead="1"
		headerLine="$line"
		printf "$headerLine\n" >> "$outname"
	elif [ $lno -eq $perFile ]
	then
		fno=$[$fno+1]
		lno="1"
		outname="$file-$fno.csv"
		printf "$headerLine\n" >> "$outname"
		printf "$line\n" >> "$outname"
	else
		printf "$line\n" >> "$outname"
		lno=$[$lno+1]
	fi
done < "$file.csv" 
