sudo apt-get install \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev


git clone https://github.com/gflags/gflags
cd gflags 
mkdir build 
cd build 
cmake -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=ON -DINSTALL_HEADERS=ON -DINSTALL_SHARED_LIBS=ON -DINSTALL_STATIC_LIBS=ON .. 
make -j24  
make install
cd ..
cd ..

export PATH=/usr/local/cuda-11.3/bin/:$PATH
git clone https://github.com/ceres-solver/ceres-solver
cd ceres-solver
mkdir build
cd build
cmake ..
make -j24 
sudo make install
cd ..
cd ..

# git clone https://github.com/google/glog.git
# cd glog
# mkdir build
# cd build/
# cmake  ..
# make -j32
# sudo make install
# cd ..
# cd ..

git clone https://github.com/colmap/colmap.git
# cd colmap
git checkout dev
mkdir build
cd build
sudo cmake .. -D CMAKE_CUDA_COMPILER="/usr/local/cuda/bin/nvcc" ../CMakeLists.txt -D CMAKE_CUDA_ARCHITECTURES='native'
cd ../
make -j32
sudo make install