import numpy as np
import unittest
import invbubble


class TestAll(unittest.TestCase):

    # def test_no_interpolate_function(self):
    #     X = np.load('xyvalues.npy')
    #     Y = np.load('dispvalues.npy')
    #     X = X.reshape(200, 937, 3)
    #     Y = Y.reshape(200, 937, 3)
    #     my_p = X[:, 0, 2]  # pressures
    #     my_int = invbubble.Interpolate(X, Y)
    #     # calc dx, dy, dz with inter
    #     p = 1e-5
    #     dx, dy, dz = my_int.calc_disp(X[0, :, :2], p)
    #     a = np.nansum(dx + dy + dz)
    #     # calc without inter
    #     p = my_p[-1]
    #     dx, dy, dz = my_int.calc_disp(X[0, :, :2], p)
    #     b = np.nansum(dx + dy + dz)
    #     print('no_interpolate', a, b)
    #     self.assertTrue( np.isclose(a, 5492.118626435472) and np.isclose(b, 18466.79512000025))  # noqa E501

    # def test_forward_interpolation_abq_data(self):
    #     X = np.load('xyvalues.npy')
    #     Y = np.load('dispvalues.npy')
    #     X = X.reshape(200, 937, 3)
    #     Y = Y.reshape(200, 937, 3)
    #     s0 = np.arange(0, 199, 2)
    #     s1 = np.arange(1, 199, 2)
    #     my_int = invbubble.Interpolate(X[s0], Y[s0])
    #     dx_delta, dy_delta, dz_delta = my_int.calc_delta(X[s1, :, :2],
    #                                                      X[s1, 0, 2],
    #                                                      Y[s1])
    #     a = np.nansum(dx_delta + dy_delta + dz_delta)
    #     print('forward', a)
    #     self.assertTrue(np.isclose(a, 0.6854286151306995))


    # def test_backward_interpolation_abq_data(self):
    #     X = np.load('xyvalues.npy')
    #     Y = np.load('dispvalues.npy')
    #     X = X.reshape(200, 937, 3)
    #     Y = Y.reshape(200, 937, 3)
    #     s1 = np.arange(2, 199, 2)
    #     s0 = np.arange(1, 200, 2)
    #     my_int = invbubble.Interpolate(X[s0], Y[s0])
    #     dx_delta, dy_delta, dz_delta = my_int.calc_delta(X[s1, :, :2],
    #                                                      X[s1, 0, 2],
    #                                                      Y[s1])
    #     b = np.nansum(dx_delta + dy_delta + dz_delta)
    #     print('backward', b)
    #     self.assertTrue(np.isclose(b, 0.23361679027349674))

    def test_on_blue_data(self):
        # # test on blue
        # blue00 = np.load('blue00.npy')
        # blue01 = np.load('blue01.npy')
        # blue02 = np.load('blue02.npy')
        # blue03 = np.load('blue03.npy')
        # p_list00 = blue00[:, 1]
        # p_list01 = blue01[:, 1]
        # p_list02 = blue02[:, 1]
        # p_list03 = blue03[:, 1]
        # bluep = np.concatenate((p_list00, p_list01, p_list02, p_list03))
        # X = np.load('xy_model.npy')
        # Y = np.load('disp_values.npy')
        # X = X.reshape(201, 937, 3)
        # Y = Y.reshape(201, 937, 3)
        # my_int = Interpolate(X, Y)
        # dx_delta00, dy_delta00, dz_delta00 = my_int.calc_delta_test(blue00[:, 0], blue00[:, 1])
        # dx_delta01, dy_delta01, dz_delta01 = my_int.calc_delta_test(blue01[:, 0], blue01[:, 1])
        # dx_delta02, dy_delta02, dz_delta02 = my_int.calc_delta_test(blue02[:, 0], blue02[:, 1])
        # dx_delta03, dy_delta03, dz_delta03 = my_int.calc_delta_test(blue03[:, 0], blue03[:, 1])
        blue00 = np.load('blue00.npy')
        p_list00 = blue00[:, 1]
        X = np.load('xy_model.npy')
        Y = np.load('disp_values.npy')
        X = X.reshape(201, 937, 3)
        Y = Y.reshape(201, 937, 3)
        my_int = invbubble.Interpolate(X, Y)
        dx_delta00, dy_delta00, dz_delta00 = my_int.calc_delta_test(blue00[:, 0], blue00[:, 1])
        print('blue data', np.nansum(dx_delta00 + dy_delta00 + dz_delta00))

# # test on black data
# blue00 = np.load('black01.npy')
# blue01 = np.load('black02.npy')
# blue02 = np.load('black03.npy')
# blue03 = np.load('black04.npy')
# p_list00 = blue00[:, 1]
# p_list01 = blue01[:, 1]
# p_list02 = blue02[:, 1]
# p_list03 = blue03[:, 1]
# blackp = np.concatenate((p_list00, p_list01, p_list02, p_list03))
# X = np.load('xy_model.npy')
# Y = np.load('disp_values.npy')
# X = X.reshape(201, 937, 3)
# Y = Y.reshape(201, 937, 3)
# my_int = Interpolate(X, Y)
# dx_delta00, dy_delta00, dz_delta00 = my_int.calc_delta_test(blue00[:, 0], blue00[:, 1])
# dx_delta01, dy_delta01, dz_delta01 = my_int.calc_delta_test(blue01[:, 0], blue01[:, 1])
# dx_delta02, dy_delta02, dz_delta02 = my_int.calc_delta_test(blue02[:, 0], blue02[:, 1])
# dx_delta03, dy_delta03, dz_delta03 = my_int.calc_delta_test(blue03[:, 0], blue03[:, 1])

if __name__ == "__main__":
    unittest.main()
