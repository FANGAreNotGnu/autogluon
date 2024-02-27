source ~/anaconda3/etc/profile.d/conda.sh
conda activate ann     # an environment that contains python and numpy

git clone https://github.com/facebookresearch/faiss.git faiss_xx

cd faiss_xx

LD_LIBRARY_PATH= MKLROOT=/private/home/matthijs/anaconda3/envs/ann/lib CXX=$(which g++) \
$cmake -B build -DBUILD_TESTING=ON -DFAISS_ENABLE_GPU=OFF \
             -DFAISS_OPT_LEVEL=axv2 \
             -DFAISS_ENABLE_C_API=ON \
             -DCMAKE_BUILD_TYPE=Release \
             -DBLA_VENDOR=Intel10_64_dyn .


make -C build -j10 swigfaiss && (cd build/faiss/python ; python3 setup.py build)

(cd tests ; PYTHONPATH=../build/faiss/python/build/lib/ OMP_NUM_THREADS=1 python -m unittest -v discover )