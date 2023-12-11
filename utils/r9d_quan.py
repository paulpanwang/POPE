import numpy as np 
from scipy.spatial.transform import Rotation as R


if __name__ == "__main__":
    Rm = np.array([[ 0.89663838 ,-0.01255485 ,-0.44258557],
                [ 0.01331523 , 0.99991038, -0.00138906],
                [ 0.44256335 ,-0.00464765,  0.8967252 ]])

    t = np.array([-2.37980519e-01, -6.49285700e-04,  9.71269711e-01])


    R1 = R.from_matrix(Rm).as_quat()
    print(R1)