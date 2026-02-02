#!/bin/bash

env_cmakelists="$(dirname $(readlink -f "$0"))/../minizero/environment/CMakeLists.txt"
support_games=($(awk '/target_include_directories/,/\)/' ${env_cmakelists} | sed 's|/|\n|g' | grep -v -E 'target|environment|PUBLIC|CMAKE_CURRENT_SOURCE_DIR|base|stochastic|)'))

usage()
{
	echo "Usage: $0 GAME_TYPE NETWORK_TYPE CONFIGURE_FILE END_ITERATION [OPTION]..."
	echo "The script for training networks."
	echo ""
	echo "Required arguments:"
    echo "  GAME_TYPE: ${support_games[@]}"
	echo "  NETWORK_TYPE: siamese/info_set_generator"
	echo "  CONFIGURE_FILE: the configure file (*.cfg) to use"
	echo "  END_ITERATION: the total number of iterations for training"
	echo ""
	echo "Optional arguments:"
	echo "  -h,        --help                 Give this help list"
	echo "  -n,        --name                 Assign name for training directory"
	echo "  -np,       --name_prefix          Add prefix name for default training directory name"
	echo "  -ns,       --name_suffix          Add suffix name for default training directory name"
	echo "  -g,        --gpu                  Assign available GPUs for worker, e.g. 0123"
	echo "  -b,        --batch_size           Assign the batch size in self-play worker (default = 64)"
	echo "  -c,        --cpu_thread_per_gpu   Assign the number of CPUs for each GPU in self-play worker (default = 4)"
	echo "  -conf_str                         Add additional configure string in self-play worker"
	echo "             --op_executable_file   Assign the path for optimization executable file"
	echo "             --link_sgf             Assign the path of sgf for training without self play (only op)"
	exit 1
}

if [ $# -lt 4 ] || [ $(($# % 2)) -ne 0 ];
then
	usage
else
	game_type=$1; shift
	network_type=$1; shift
	configure_file=$1; shift
	end_iteration=$1; shift
	
	# default arguments
	num_gpu=$(nvidia-smi -L | wc -l)
	gpu_list=$(echo $num_gpu | awk '{for(i=0;i<$1;i++)printf i}')
	batch_size=64
	max_num_cpu_thread_per_gpu=4
	overwrite_conf_str=""
fi

train_dir=""
name_prefix=""
name_suffix=""
link_sgf=""
sp_executable_file=build/${game_type}/minizero_${game_type}

if [[ ${network_type} == "siamese" ]]; then
	op_executable_file=minizero/learner/train_siamese.py
elif [[ ${network_type} == "info_set_generator" ]]; then
	op_executable_file=minizero/learner/train_info_set_generator.py
fi

while :; do
	case $1 in
		-h|--help) shift; usage
		;;
		-n|--name) shift; train_dir=$1
		;;
		-np|--name_prefix) shift; name_prefix=$1
		;;
		-ns|--name_suffix) shift; name_suffix=$1
		;;
		-g|--gpu) shift; gpu_list=$1; num_gpu=${#gpu_list}
		;;
		-b|--batch_size) shift; batch_size=$1
		;;
		-c|--cpu_thread_per_gpu) shift; max_num_cpu_thread_per_gpu=$1
		;;
		-conf_str) shift; overwrite_conf_str=":$1"
		;;
		--link_sgf) shift; link_sgf=$1
		;;
		--op_executable_file) shift; op_executable_file=$1
		;;
		"") break
		;;
		*) echo "Unknown argument: $1"; usage
		;;
	esac
	shift
done

# create default name; also check if configurations are valid
testrun_stderr_tmp=$(mktemp)
default_name=$(${sp_executable_file} -mode zero_training_name -conf_file ${configure_file} -conf_str "${overwrite_conf_str}" 2>${testrun_stderr_tmp} || :)
testrun_stderr=$(<${testrun_stderr_tmp})
rm -f ${testrun_stderr_tmp}
if [[ ! ${default_name} ]]; then
	echo "${testrun_stderr}" >&2
	exit 1
fi
# use default name of training if name is not assigned
if [[ -z ${train_dir} ]]; then
	train_dir=${name_prefix}${default_name}${name_suffix}
fi

# arguments
cuda_devices=$(echo ${gpu_list} | awk '{ split($0, chars, ""); printf(chars[1]); for(i=2; i<=length(chars); ++i) { printf(","chars[i]); } }')
max_num_cpu_thread=$((max_num_cpu_thread_per_gpu*num_gpu))
num_cpu_thread=$(lscpu -p=CORE | grep -v "#" | wc -l)
if [ $num_cpu_thread -gt $max_num_cpu_thread ]; then
	num_cpu_thread=$max_num_cpu_thread
fi

run_stage="R"
if [ -d ${train_dir} ]; then
	read -n1 -p "${train_dir} has existed. (R)estart / (C)ontinue / (Q)uit? " run_stage
	echo ""
fi

if [[ ${run_stage,} == "r" ]]; then
	rm -rf ${train_dir}
	echo "create ${train_dir} ..."
	mkdir -p ${train_dir}/model ${train_dir}/sgf
	if [[ ! -z ${link_sgf} ]];
	then
		ln ${link_sgf}/* ${train_dir}/sgf/
		end_iteration=$(ls ${train_dir}/sgf/ | wc -l)
		echo "link ${link_sgf} ..."
		echo "end_iteration: ${end_iteration}"
	fi
	touch ${train_dir}/Training.log
	touch ${train_dir}/op.log
	new_configure_file=$(basename ${train_dir}).cfg
	${sp_executable_file} -gen ${train_dir}/${new_configure_file} -conf_file ${configure_file} -conf_str "${overwrite_conf_str}" 2>/dev/null

	# setup initial weight
	cuda_devices=$(echo ${gpu_list} | awk '{ split($0, chars, ""); printf(chars[1]); for(i=2; i<=length(chars); ++i) { printf(","chars[i]); } }')
	echo "train \"\" -1 -1" | CUDA_VISIBLE_DEVICES=${cuda_devices} PYTHONPATH=. python ${op_executable_file} ${game_type} ${train_dir} ${train_dir}/${new_configure_file} >/dev/null 2>&1
	echo -e "start\ntrain weight_iter_0.pkl 1 1" | CUDA_VISIBLE_DEVICES=${cuda_devices} PYTHONPATH=. python ${op_executable_file} ${game_type} ${train_dir} ${train_dir}/${new_configure_file} 2> >(tee -a ${train_dir}/op.log >&2)
else
	exit
fi
