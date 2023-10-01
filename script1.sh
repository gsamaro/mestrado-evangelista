#PBS -N mestrado_first_formulation
#PBS -q paralela
#PBS -l nodes=8:ppn=128
#PBS -e outputs/erros1
#PBS -o outputs/saidas1
#PBS -m abe
#PBS -M gabriel.amaro94@gmail.com

module load python/3.8.11-intel-2021.3.0
module load openmpi/4.1.1-intel-2021.3.0
source py38/bin/activate

cd $PBS_O_WORKDIR
mpirun python -m mpi4py.futures src/first_formulation.py