#PBS -N main
#PBS -q memshort
#PBS -l nodes=1:ppn=128
#PBS -e outputs/erros_memshort
#PBS -o outputs/saidas_memshort
#PBS -m ae
#PBS -M gabriel.amaro94@gmail.com

module load python/3.8.11-intel-2021.3.0
module load openmpi/4.1.1-intel-2021.3.0
source py38/bin/activate

cd $PBS_O_WORKDIR
mpirun python -m mpi4py.futures src/main.py