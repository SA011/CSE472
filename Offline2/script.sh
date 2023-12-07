dataset_no=1
if [[ $# -eq 1 ]]; then
    dataset_no=$1
fi

echo "Dataset no: $dataset_no"

if [[ $dataset_no -eq 1 ]]; then
python3 ./solve.py --dataset_no 1 \
                    --input_file 'dataset1/WA_Fn-UseC_-Telco-Customer-Churn.csv' \
                    --epochs 1000 \
                    --number_of_learners 20 \
                    --feature_count 15 \
                    --mini_batch_size 100 \
                    --missing_value 'mean' \
                    --learning_rate 100 \
                    --k_fold 1 \
                    --dataset_size 10000 \
                    --seed 1
fi

if [[ $dataset_no -eq 2 ]]; then
python3 ./solve.py --dataset_no 2 \
                    --input_file 'dataset2/adult.data$dataset2/adult.test' \
                    --epochs 100 \
                    --number_of_learners 20 \
                    --feature_count 20 \
                    --mini_batch_size 100 \
                    --missing_value 'mean' \
                    --learning_rate 100 \
                    --k_fold 1 \
                    --dataset_size 200000 
fi

if [[ $dataset_no -eq 3 ]]; then
python3 ./solve.py --dataset_no 3 \
                    --input_file 'dataset3/creditcard.csv' \
                    --epochs 1000 \
                    --number_of_learners 20 \
                    --feature_count 10 \
                    --mini_batch_size 100 \
                    --missing_value 'mean' \
                    --learning_rate 100 \
                    --k_fold 1 \
                    --dataset_size 20000 \
                    --all_positive True
fi