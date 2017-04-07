for f in *.txt; do
	python main.py -f "$f" >> output.txt;
done

awk 'NR % 2 == 0' output.txt >> clean_output.txt;
