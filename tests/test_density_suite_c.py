"""Additional density tests for test:lib ratio on domain-heavy repos."""
from __future__ import annotations
import numpy as np

def test_density_c_00():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=0)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_01():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=1)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_02():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=2)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_03():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=3)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_04():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=4)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_05():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=5)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_06():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=6)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_07():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=7)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_08():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=8)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_09():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=9)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_10():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=10)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_11():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=11)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_12():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=12)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_13():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=13)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_14():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=14)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_15():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=15)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_16():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=16)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_17():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=17)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_18():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=18)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_19():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=19)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_20():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=20)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_21():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=21)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_22():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=22)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_23():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=23)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_24():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=24)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_25():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=25)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_26():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=26)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_27():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=27)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_28():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=28)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_29():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=29)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_30():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=30)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_31():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=31)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_32():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=32)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_33():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=33)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_34():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=34)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_35():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=35)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_36():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=36)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_37():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=37)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_38():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 0])
    p = np.array([0, 1, 0, 1, 0])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=38)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

def test_density_c_39():
    from headroom.evaluation.metrics import accuracy, f1, bootstrap_mean
    y = np.array([0, 1, 0, 1, 1])
    p = np.array([0, 1, 0, 1, 1])
    assert accuracy(y, p) == 1.0
    assert 0.0 <= f1(y, p) <= 1.0
    est = bootstrap_mean(y.astype(float), n_boot=50, seed=39)
    assert est.n == 5
    assert est.lo <= est.value <= est.hi

