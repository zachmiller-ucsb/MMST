if [ $# != 1 ]; then
    echo "Usage: gen_test.py gen_input"
    exit 1
fi

python3 gen_test.py < $1 > gen.txt
python3 2approx.py < gen.txt > out
python3 MBMST.py < gen.txt >> out
