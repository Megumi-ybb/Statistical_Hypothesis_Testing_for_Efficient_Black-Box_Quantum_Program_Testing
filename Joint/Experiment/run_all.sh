#!/usr/bin/env bash

files=(
  joint_experiemnt_BVtestfalse.py
  joint_experiemnt_BVtestfalse_falsecase.py
  joint_experiemnt_grovertestfalse.py
  joint_experiemnt_grovertestfalse_falsecase.py
  joint_addtestfalse.py
  joint_addtestfalse_falsecase.py
  joint_QFTtestfalse.py
  joint_QFTtestfalse_falsecase.py
  joint_QPEtestfalse.py
  joint_QPEtestfalse_falsecase.py
  joint_QWALKtestfalse.py
  joint_QWALKtestfalse_falsecase.py
)

for nq in {2..9}; do
  for f in "${files[@]}"; do
    echo "Submitting $f with n_qubits=$nq"
    grid_run --grid_submit=batch --grid_ncpus=100 --grid_mem=250G "./$f" "$nq"
  done
done

