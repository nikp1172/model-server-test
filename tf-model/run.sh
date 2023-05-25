MODELS_DIR='.'
MLMODEL=$(find $MODELS_DIR -type f -name MLmodel -print | head -n 1)
PYTHON_VERSION=$(yq .flavors.python_function.python_version $MLMODEL | tr -d '"' | awk -F. '{ print $1"."$2 }')
conda create --name serve-env -y python=$PYTHON_VERSION
conda activate serve-env
REQ_TXT=$(find $MODELS_DIR -type f -name requirements.txt -print | head -n 1)
grep -E -v 'tfy-mlflow-client|mlfoundry' $REQ_TXT > .filtered_requirements.txt
pip install -U pip
echo "Installing model pip package requirements:-"
cat .filtered_requirements.txt
pip install -r --use-deprecated=legacy-resolver .filtered_requirements.txt mlflow==1.26.0 mlserver==1.1.0 mlserver-mlflow==1.1.0 "protobuf<=3.20.0"
mlserver start $MODELS_DIR