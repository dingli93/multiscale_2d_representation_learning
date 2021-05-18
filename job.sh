cd /running_package
hdfs dfs -get hdfs://hobot-bigdata/user/ding01.li/dataset/.vector_cache
mv /running_package/.vector_cache /running_package/2d_tan/
ls ./2d_tan
hdfs dfs -get hdfs://hobot-bigdata/user/ding01.li/tools/anaconda3.tar.gz
tar zxf anaconda3.tar.gz
hdfs dfs -get hdfs://hobot-bigdata/user/ding01.li/dataset/TACoS
cd ${WORKING_PATH}
CWD=`pwd`
ls
nvidia-smi
which python
cd 2d_tan
CMD="/running_package/mnt/data-8/data/ding.li/anaconda3/bin/python /running_package/2d_tan/moment_localization/train.py --cfg experiments/tacos/2D-TAN-128x128-K5L8-pool.yaml --verbose"
echo Running ${CMD}
${CMD}
