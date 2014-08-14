# python run_test.py
# test all test_set
#
# python run_test.py [keyword]
# test test_set which involve keyword

import os, sys

test_list = []
# query_2d
test_list.append('0_2D/query_2d/query_2d.vvl')
test_list.append('0_2D/replicate.vvl')
test_list.append('0_2D/reflect.vvl')
test_list.append('0_2D/reflect_101.vvl')
test_list.append('0_2D/wrap.vvl')

# query_3d
test_list.append('1_3D/3d.vvl')
test_list.append('1_3D/query_3d.vvl')
test_list.append('1_3D/reflect.vvl')
test_list.append('1_3D/reflect_101.vvl')
test_list.append('1_3D/wrap.vvl')

# iterator
test_list.append('2_Iterator/line_iter.vvl')
test_list.append('2_Iterator/plane_iter.vvl')
test_list.append('2_Iterator/cube_iter.vvl')
test_list.append('2_Iterator/cube_iter_count.vvl')
test_list.append('2_Iterator/orthogonal.vvl')
test_list.append('2_Iterator/orthogonal_ones.vvl')


# input split
test_list.append('4_Parallelization/input_split/input_split_x_axis.vvl')
test_list.append('4_Parallelization/input_split/input_split_y_axis.vvl')
test_list.append('4_Parallelization/input_split/mat_mul_split_xy.vvl')
test_list.append('4_Parallelization/input_split/mat_mul_split_yx.vvl')

# output split
test_list.append('4_Parallelization/output_split/output_split_x2.vvl')
test_list.append('4_Parallelization/output_split/output_split_x5.vvl')
test_list.append('4_Parallelization/output_split/output_split_x3y3.vvl')

# in_and_out split
test_list.append('4_Parallelization/in_and_out_split/in_and_out_split_x2.vvl')
test_list.append('4_Parallelization/in_and_out_split/in_and_out_split_x5.vvl')
test_list.append('4_Parallelization/in_and_out_split/in_and_out_split_x3y3.vvl')
test_list.append('4_Parallelization/in_and_out_split/in_and_out_split_x2x2.vvl')
test_list.append('4_Parallelization/in_and_out_split/in_and_out_split_x2x2_diff.vvl')
test_list.append('4_Parallelization/in_and_out_split/mat_mul_split_yx.vvl')

# halo
test_list.append('5_Halo/in_halo.vvl')
test_list.append('5_Halo/out_halo.vvl')

# constant
test_list.append('6_Constant/constant1.vvl')
test_list.append('6_Constant/constant4.vvl')
test_list.append('6_Constant/constant5.vvl')
test_list.append('6_Constant/constant6.vvl')
test_list.append('6_Constant/constant7.vvl')

# dtype test
test_list.append('7_Dtype/uchar.vvl')
test_list.append('7_Dtype/char.vvl')
test_list.append('7_Dtype/short.vvl')
test_list.append('7_Dtype/ushort.vvl')
test_list.append('7_Dtype/int.vvl')
test_list.append('7_Dtype/uint.vvl')
test_list.append('7_Dtype/float.vvl')
test_list.append('7_Dtype/double.vvl')

# Modifier
test_list.append('8_Modifier/dtype/dtype.vvl')
test_list.append('8_Modifier/execid/execid.vvl')
test_list.append('8_Modifier/halo/halo.vvl')
test_list.append('8_Modifier/merge/merge.vvl')
test_list.append('8_Modifier/range/range.vvl')
test_list.append('8_Modifier/split/split.vvl')


n = len(sys.argv)
if n >= 2:
	keyword = sys.argv[1]
else:
	keyword = ''

count = 1
n = 0
# count number of total test will be executed
for test in test_list:
	if keyword in test:
		n += 1

# execute test
for test in test_list:
	if keyword in test:
		print "TEST", test, 'Progress:', count, '/', n
		cmd = 'vivaldi ' + test
		e = os.system(cmd)
		count += 1


