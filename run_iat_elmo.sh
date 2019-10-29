date=`date '+%Y%m%d-%H%M%S'`
models="elmo"
data_dir="data/iats"
exp="word"
result_dir="results/${date}_${exp}_${models}_${model_type}"
log_file="${result_dir}/log.log"
results_path="${result_dir}/results.tsv"
exp_dir=${result_dir}
# tests="weat1,weat2,weat3,weat3b,weat4,weat5,weat5b,weat6,weat6b,weat7,weat7b,weat8,weat8b,weat9,weat10,weat_hdb_competent,weat_hdb_likable,weat_angry_black_woman_stereotype,weat_angry_black_woman_stereotype_b,weat_r_hdb_competent,weat_r_hdb_likable,weat+11,weat+12,weat+13"
tests="weat+occ,weat+i1,weat+i2,weat+i3,weat+i4,weat+i5"
seed=1111
export CUDA_VISIBLE_DEVICES=5
python iat.py \
    --tests ${tests} \
    --exp ${exp} \
    --models ${models} \
    --seed ${seed} \
    --log_file ${log_file} \
    --results_path ${results_path} \
    --data_dir ${data_dir} \
    --exp_dir ${exp_dir} &

date=`date '+%Y%m%d-%H%M%S'`
models="elmo"
data_dir="data/iats"
exp="sent"
result_dir="results/${date}_${exp}_${models}_${model_type}"
log_file="${result_dir}/log.log"
results_path="${result_dir}/results.tsv"
exp_dir=${result_dir}
# tests="sent-weat1,sent-weat2,sent-weat3,sent-weat3b,sent-weat4,sent-weat5,sent-weat5b,sent-weat6,sent-weat6b,sent-weat7,sent-weat7b,sent-weat8,sent-weat8b,sent-weat9,sent-weat10,sent-weat_hdb_bleach_competent,sent-weat_hdb_bleach_likable,sent-weat_hdb_unbleach_competent,sent-weat_hdb_unbleach_likable,sent-weat_angry_black_woman_stereotype,sent-weat_angry_black_woman_stereotype_b,sent-weat_r_hdb_bleach_competent,sent-weat_r_hdb_bleach_likable,sent-weat_r_hdb_unbleach_competent,sent-weat_r_hdb_unbleach_likable,sent-weat+11,sent-weat+12,sent-weat+13"
tests="sent-weat+occ,sent-weat+i1,sent-weat+i2,sent-weat+i3,sent-weat+i4,sent-weat+i5"
seed=1111
export CUDA_VISIBLE_DEVICES=5
python iat.py \
    --tests ${tests} \
    --exp ${exp} \
    --models ${models} \
    --seed ${seed} \
    --log_file ${log_file} \
    --results_path ${results_path} \
    --data_dir ${data_dir} \
    --exp_dir ${exp_dir} &

date=`date '+%Y%m%d-%H%M%S'`
models="elmo"
data_dir="data/iats"
exp="c-word"
result_dir="results/${date}_${exp}_${models}_${model_type}"
log_file="${result_dir}/log.log"
results_path="${result_dir}/results.tsv"
exp_dir=${result_dir}
# tests="sent-weat1,sent-weat2,sent-weat3,sent-weat3b,sent-weat4,sent-weat5,sent-weat5b,sent-weat6,sent-weat6b,sent-weat7,sent-weat7b,sent-weat8,sent-weat8b,sent-weat9,sent-weat10,sent-weat_hdb_bleach_competent,sent-weat_hdb_bleach_likable,sent-weat_hdb_unbleach_competent,sent-weat_hdb_unbleach_likable,sent-weat_angry_black_woman_stereotype,sent-weat_angry_black_woman_stereotype_b,sent-weat_r_hdb_bleach_competent,sent-weat_r_hdb_bleach_likable,sent-weat_r_hdb_unbleach_competent,sent-weat_r_hdb_unbleach_likable,sent-weat+11,sent-weat+12,sent-weat+13"
tests="sent-weat+occ,sent-weat+i1,sent-weat+i2,sent-weat+i3,sent-weat+i4,sent-weat+i5"
seed=1111
export CUDA_VISIBLE_DEVICES=5
python iat.py \
    --tests ${tests} \
    --exp ${exp} \
    --models ${models} \
    --seed ${seed} \
    --log_file ${log_file} \
    --results_path ${results_path} \
    --data_dir ${data_dir} \
    --exp_dir ${exp_dir} &
