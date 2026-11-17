const int K = xq_shape[1];
const int M = xq_shape[0];
const int N = wq_shape[0];
const int m0 = int(threadgroup_position_in_grid.y) * BM;
const int n0 = int(threadgroup_position_in_grid.x) * BN;
constexpr auto desc = matmul2d_descriptor(BM, BN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc, execution_simdgroups<4>> op;
tensor<device int8_t, dextents<int, 2>, tensor_inline> A((device int8_t*)xq, dextents<int, 2>(K, M));
tensor<device int8_t, dextents<int, 2>, tensor_inline> B((device int8_t*)wq, dextents<int, 2>(K, N));
auto a0 = A.template slice<BK, BM>(0, m0);
auto b0 = B.template slice<BK, BN>(0, n0);
auto cT = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), int32_t>();
for (uint16_t i = 0; i < cT.get_capacity(); ++i) cT[i] = 0;
for (int k0 = 0; k0 < K; k0 += BK) {
  auto a = A.template slice<BK, BM>(k0, m0);
  auto b = B.template slice<BK, BN>(k0, n0);
  op.run(a, b, cT);
}
