#PBS -N mestrado_zero_formulation
#PBS -q parexp
#PBS -l nodes=22:ppn=48
#PBS -e outputs/erros0
#PBS -o outputs/saidas0
#PBS -m abe
#PBS -M gabriel.amaro94@gmail.com

module load python/3.8.11-intel-2021.3.0
module load openmpi/4.1.1-intel-2021.3.0
source py38/bin/activate

cd $PBS_O_WORKDIR
mpirun python -m mpi4py.futures src/first_formulation.py