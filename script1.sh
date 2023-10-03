#PBS -N main
#PBS -q paralela
#PBS -l nodes=3:ppn=128
#PBS -e outputs/erros1
#PBS -o outputs/saidas1
#PBS -m ae
#PBS -M gabriel.amaro94@gmail.com

module load python/3.8.11-intel-2021.3.0
module load openmpi/4.1.1-intel-2021.3.0
source py38/bin/activate

cd $PBS_O_WORKDIR
mpirun python -m mpi4py.futures src/main.py