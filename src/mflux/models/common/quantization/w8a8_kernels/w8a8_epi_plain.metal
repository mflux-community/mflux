auto dT = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), T>();
for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
  auto ids = cT.get_multidimensional_index(i);
  const int n = min(n0 + ids[0], N - 1), m = min(m0 + ids[1], M - 1);
  dT[i] = T(float(cT[i]) * xs[m] * ws[n]);
}
if (n0 == n_lo && m0 + BM <= M) {
  tensor<device T, dextents<int, 2>, tensor_inline> D((device T*)y, dextents<int, 2>(N, M));
  auto dS = D.template slice<BN, BM>(n0, m0);
  dT.store(dS);
} else {
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    auto ids = cT.get_multidimensional_index(i);
    const int n = n0 + ids[0], m = m0 + ids[1];
    if (m < M && n >= n_lo && n < N) y[(size_t)m * N + n] = dT[i];
  }
}
