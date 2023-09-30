#PBS -N mestrado_first_formulation
#PBS -q testes
#PBS -l nodes=1:ppn=2
#PBS -e erros
#PBS -o saida
#PBS -m abe
#PBS -M gabriel.amaro94@gmail.com

module load python/3.8.11-intel-2021.3.0
module load openmpi/4.1.1-intel-2021.3.0
source py38/bin/activate

cd $PBS_O_WORKDIR
mpirun -np 4 python -m mpi4py.futures mestrado-evangelista/src/first_formulation.py