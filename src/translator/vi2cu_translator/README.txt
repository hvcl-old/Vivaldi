How to Test

python run_test.py [function_name]

function_name:
1. divide_line
2. code_to_line_list
3. dtype_check
4. split_into_block_and_code
5. to_CUDA_style

example 
$ python run_test.py divide_line

-----------------------------------------------------------------------------

vi2cu compiler structure
vi2cu_compiler/
	__init__.py
	main.py
	misc.py
	run_test.py
	README.txt
	function directory/
		code_to_line_list/
			__init__.py
			code_to_line_list.py
			test_set/
				test_list
				...
		divide_line/
			__init__.py
			divide_line.py
			test_set/
				test_list
				...
		dtype_check/
			__init__.py
			dtype_check.py
			test_set/
				test_list
				...
		split_into_block_and_code/
			__init__.py
			split_into_block_and_code.py
			test_set/
				test_list
				...
		to_CUDA_style/
			__init__.py
			to_CUDA_style.py
			test_set/
				test_list
				...