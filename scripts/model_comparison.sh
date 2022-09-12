HYDRA_FULL_ERROR=1 python -u train_mlgapfill.py -m \
experiment_name='classifier_comparison' \
run_idx=1,2,3,4,5,6,7 \
model=svm_classifier \
model.C=0.1,0.25,0.5,0.75,1.0,5.0,10.0,25.0,50.0,100.0,1000.0 \
model.kernel='rbf','poly' \
target_Y.0='Y_growth_class' \
test_size=0.2


HYDRA_FULL_ERROR=1 python -u train_mlgapfill.py -m \
experiment_name='classifier_comparison' \
run_idx=1 \
model=svm_classifier \
model.C=0.75 \
model.kernel='rbf' \
target_Y.0='Y_growth_class' \
test_size=0.2


python -u train_mlgapfill.py -m \
experiment_name='train_random_forest' \
run_idx=1,2,3,4,5 \
target_Y.0='Y_growth_class' \
model=random_forest_classifier \
model.n_estimators=100,250,500 \
model.max_depth=null,10,50,100 \
model.max_features=sqrt \
test_size=0.2

HYDRA_FULL_ERROR=1 python -u train_mlgapfill.py -m \
experiment_name='classifier_comparison' \
run_idx=1 \
target_Y.0='Y_growth_class' \
model=random_forest_classifier \
model.n_estimators=50 \
model.max_depth=10 \
model.max_features=sqrt \
test_size=0.3


python -u train_mlgapfill.py -m \
experiment_name='random_forest' \
run_idx=1,2,3,4,5 \
target_Y.0='Y_growth_class' \
model=random_forest_classifier \
model.n_estimators=250 \
model.max_depth=100 \
model.max_features=sqrt \
test_size=0.5



python -u train_mlgapfill.py -m \
experiment_name='random_forest' \
run_idx=1,2,3,4,5,6,7,8,9,10 \
target_Y.0='Y_growth_class' \
model=random_forest_classifier \
model.n_estimators=250 \
model.max_depth=10 \
model.max_features=sqrt \
test_size=0.8
