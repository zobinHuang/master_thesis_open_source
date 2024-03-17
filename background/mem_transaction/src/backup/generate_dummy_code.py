print(f"#define TEST_L1 \\")
print(f"    test_l1<dtype, 1, num_iters>(f); \\")
print(f"    test_l1<dtype, 2, num_iters>(f); \\")
print(f"    test_l1<dtype, 4, num_iters>(f); \\")
print(f"    test_l1<dtype, 8, num_iters>(f); \\")

i=16
while i <= 2**15:
    if i != 2**15:
        print(f"    test_l1<dtype, {i}, num_iters>(f); \\")
    else:
        print(f"    test_l1<dtype, {i}, num_iters>(f);")
    i += 8  # 8 * 4 = 32