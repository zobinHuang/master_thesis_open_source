sudo docker run --gpus all -dit --privileged -v $PWD:/root --name zobin_ncu_profile zobinhuang/pos_svr_base:11.3
sudo docker exec -it zobin_ncu_profile bash
# sudo docker container stop zobin_ncu_profile
# sudo docker container rm zobin_ncu_profile
