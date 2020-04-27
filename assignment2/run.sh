make clean
make

./dt.exe data/dt_train1.txt data/dt_test1.txt result/dt_result1.txt 

./test/dt_test.exe test/dt_answer1.txt result/dt_result1.txt
