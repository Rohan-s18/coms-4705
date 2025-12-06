#!/bin/bash

# Compile C programs
echo "Compiling C programs..."
gcc -O3 -Wall -o dp1 dp1.c
gcc -O3 -Wall -o dp2 dp2.c
gcc -O3 -Wall -o dp3 dp3.c -lmkl_rt

# Define test cases
declare -a Nvals=("1000000" "300000000")
declare -a Reps=("1000" "20")

# Run dp1.c
echo "Running dp1..."
for i in ${!Nvals[@]}; do
    ./dp1 ${Nvals[$i]} ${Reps[$i]}
done

# Run dp2.c
echo "Running dp2..."
for i in ${!Nvals[@]}; do
    ./dp2 ${Nvals[$i]} ${Reps[$i]}
done

# Run dp3.c (MKL)
echo "Running dp3..."
for i in ${!Nvals[@]}; do
    ./dp3 ${Nvals[$i]} ${Reps[$i]}
done

# Run dp4.py (Python loop)
echo "Running dp4.py..."
for i in ${!Nvals[@]}; do
    python3 dp4.py ${Nvals[$i]} ${Reps[$i]}
done

# Run dp5.py (NumPy dot)
echo "Running dp5.py..."
for i in ${!Nvals[@]}; do
    python3 dp5.py ${Nvals[$i]} ${Reps[$i]}
done
