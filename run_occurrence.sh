# evaluation
result_dir="results/occurrence"
female_stereo_occs="/home/lily/yt325/150/project/data/winobias/female_occupations.txt"
male_stereo_occs="/home/lily/yt325/150/project/data/winobias/male_occupations.txt"

# dataset="1bword"
# data_path="/home/lily/yt325/150/project/data/1bword/training-monolingual.tokenized.shuffled"
# python occurrence.py \
#     --dataset ${dataset} \
#     --data_path ${data_path} \
#     --f_att_tgt ${female_stereo_occs} \
#     --m_att_tgt ${male_stereo_occs} \
#     > ${result_dir}/${dataset}.log 2>&1 &

# dataset="wikipedia"
# data_path="/home/lily/yt325/150/project/data/wikipedia/output"
# python occurrence.py \
#     --dataset ${dataset} \
#     --data_path ${data_path} \
#     --f_att_tgt ${female_stereo_occs} \
#     --m_att_tgt ${male_stereo_occs} \
#     > ${result_dir}/${dataset}.log 2>&1 &

# dataset="bookcorpus"
# data_path="/home/lily/yt325/150/project/data/bookcorpus/bookcorpus.txt"
# python occurrence.py \
#     --dataset ${dataset} \
#     --data_path ${data_path} \
#     --f_att_tgt ${female_stereo_occs} \
#     --m_att_tgt ${male_stereo_occs} \
#     > ${result_dir}/${dataset}.log 2>&1 &
#
dataset="webtext"
data_path="/home/lily/yt325/150/project/data/webtext/webtext.train.jsonl"
python occurrence.py \
    --dataset ${dataset} \
    --data_path ${data_path} \
    --f_att_tgt ${female_stereo_occs} \
    --m_att_tgt ${male_stereo_occs} \
    > ${result_dir}/${dataset}.log 2>&1 &
