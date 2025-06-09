#!/bin/bash
# Check and edit all options before the first run!
# While reading is fine, please dont write anything to the default directories in this script

# Start and end for resamples
max_folds=5
start_fold=1

# To avoid hitting the cluster queue limit we have a higher level queue
max_num_submitted=100

# Queue options are https://sotonac.sharepoint.com/teams/HPCCommunityWiki/SitePages/Iridis%205%20Job-submission-and-Limits-Quotas.aspx
queue="ecsstaff,ecsall"

# Enter your username and email here
username="cq2u24"
mail="ALL"
mailto="$username@soton.ac.uk"

# MB for jobs, increase incrementally and try not to use more than you need. If you need hundreds of GB consider the huge memory queue
max_memory=30000

# Max allowable is 60 hours
max_time="60:00:00"


gpus=1

# Start point for the script i.e. 3 datasets, 3 classifiers = 9 jobs to submit, start_point=5 will skip to job 5
start_point=1

# Put your home directory here
local_path="/home/$username/"

script_file_path="$local_path/MAML-Pytorch/no_blur_train.py"
env_name="/scratch/cq2u24/conda-envs/meta_learning_env"
PYTHON_EXECUTABLE="${env_name}/bin/python"

echo "#!/bin/bash
#SBATCH -A ecsstaff
#SBATCH --gres=gpu:${gpus}
#SBATCH --mail-type=${mail}
#SBATCH --mail-user=${mailto}
#SBATCH --partition=${queue}
#SBATCH -t ${max_time}
#SBATCH --job-name=meta_learning
#SBATCH --mem=${max_memory}M
#SBATCH -c 4
#SBATCH --nodes=1
#SBATCH -o log/%A-%a.out # 添加这行：标准输出将写入此文件
#SBATCH -e log/%A-%a.err # 添加这行：标准错误将写入此文件

. /etc/profile
module purge
module load anaconda/py3.10
source activate ${env_name}

python -u ${script_file_path}"  > generatedFile.sub

sbatch < generatedFile.sub

echo Finished submitting jobs
