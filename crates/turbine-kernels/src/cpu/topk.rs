//! Top-k selection exactly as PyTorch's CPU `torch.topk` makes it (largest, float32).
//!
//! transformers' MoE routers pick experts with `torch.topk` over the softmax weights, and with
//! BF16 router logits exact ties are common. PyTorch (`aten/src/ATen/native/TopKImpl.h`) fills
//! a `(value, index)` array in index order and selects with libstdc++ `std::nth_element`
//! (introselect) at `k − 1`, or `std::partial_sort` (heap select) when `k · 64 ≤ n`, comparing
//! only values (NaN above everything). Which of several tied entries lands inside the top `k`
//! is therefore a property of those algorithms, not "the lower index first". This module is a
//! line-by-line port of the libstdc++ algorithms involved, so the selected set equals
//! PyTorch's; `torch_topk_fixture.txt` pins it to `torch.topk` itself
//! (`scripts/golden/torch_topk_fixture.py`).

/// One array element: the value and its original index.
type Elem = (f32, u32);

/// PyTorch's comparator for `largest`: NaN sorts first, then greater values.
fn greater(x: &Elem, y: &Elem) -> bool {
    (x.0.is_nan() && !y.0.is_nan()) || x.0 > y.0
}

/// `std::__move_median_to_first(result, a, b, c)`.
fn move_median_to_first(v: &mut [Elem], result: usize, a: usize, b: usize, c: usize) {
    let pick = if greater(&v[a], &v[b]) {
        if greater(&v[b], &v[c]) {
            b
        } else if greater(&v[a], &v[c]) {
            c
        } else {
            a
        }
    } else if greater(&v[a], &v[c]) {
        a
    } else if greater(&v[b], &v[c]) {
        c
    } else {
        b
    };
    v.swap(result, pick);
}

/// `std::__unguarded_partition(first, last, pivot)`.
fn unguarded_partition(v: &mut [Elem], mut first: usize, mut last: usize, pivot: usize) -> usize {
    loop {
        while greater(&v[first], &v[pivot]) {
            first += 1;
        }
        last -= 1;
        while greater(&v[pivot], &v[last]) {
            last -= 1;
        }
        if first >= last {
            return first;
        }
        v.swap(first, last);
        first += 1;
    }
}

/// `std::__unguarded_partition_pivot(first, last)`.
fn partition_pivot(v: &mut [Elem], first: usize, last: usize) -> usize {
    let mid = first + (last - first) / 2;
    move_median_to_first(v, first, first + 1, mid, last - 1);
    unguarded_partition(v, first + 1, last, first)
}

/// `std::__insertion_sort(first, last)`.
fn insertion_sort(v: &mut [Elem], first: usize, last: usize) {
    if first == last {
        return;
    }
    for i in first + 1..last {
        let val = v[i];
        if greater(&val, &v[first]) {
            v.copy_within(first..i, first + 1);
            v[first] = val;
        } else {
            // std::__unguarded_linear_insert
            let mut j = i;
            while greater(&val, &v[j - 1]) {
                v[j] = v[j - 1];
                j -= 1;
            }
            v[j] = val;
        }
    }
}

/// `std::__adjust_heap(first, hole, len, value)` followed by its `std::__push_heap`.
fn adjust_heap(v: &mut [Elem], first: usize, mut hole: usize, len: usize, value: Elem) {
    let top = hole;
    let mut child = hole;
    while len > 0 && child < (len - 1) / 2 {
        child = 2 * (child + 1);
        if greater(&v[first + child], &v[first + child - 1]) {
            child -= 1;
        }
        v[first + hole] = v[first + child];
        hole = child;
    }
    if (len & 1) == 0 && child == (len - 2) / 2 {
        child = 2 * (child + 1);
        v[first + hole] = v[first + child - 1];
        hole = child - 1;
    }
    while hole > top {
        let parent = (hole - 1) / 2;
        if !greater(&v[first + parent], &value) {
            break;
        }
        v[first + hole] = v[first + parent];
        hole = parent;
    }
    v[first + hole] = value;
}

/// `std::__make_heap(first, last)`.
fn make_heap(v: &mut [Elem], first: usize, last: usize) {
    let len = last - first;
    if len < 2 {
        return;
    }
    let mut parent = (len - 2) / 2;
    loop {
        let value = v[first + parent];
        adjust_heap(v, first, parent, len, value);
        if parent == 0 {
            return;
        }
        parent -= 1;
    }
}

/// `std::__heap_select(first, middle, last)`: leaves the "greatest" `middle − first` elements
/// in `[first, middle)`.
fn heap_select(v: &mut [Elem], first: usize, middle: usize, last: usize) {
    make_heap(v, first, middle);
    for i in middle..last {
        if greater(&v[i], &v[first]) {
            // std::__pop_heap(first, middle, i)
            let value = v[i];
            v[i] = v[first];
            adjust_heap(v, first, 0, middle - first, value);
        }
    }
}

/// `std::nth_element(v.begin(), v.begin() + nth, v.end())` (`std::__introselect`).
fn nth_element(v: &mut [Elem], nth: usize) {
    let (mut first, mut last) = (0usize, v.len());
    if first == last || nth == last {
        return;
    }
    // std::__lg(n) * 2
    let mut depth = 2 * (usize::BITS - 1 - last.leading_zeros());
    while last - first > 3 {
        if depth == 0 {
            heap_select(v, first, nth + 1, last);
            v.swap(first, nth);
            return;
        }
        depth -= 1;
        let cut = partition_pivot(v, first, last);
        if cut <= nth {
            first = cut;
        } else {
            last = cut;
        }
    }
    insertion_sort(v, first, last);
}

/// The indices of the `k` entries `torch.topk(values, k)` selects on the CPU (float32,
/// `largest=True`), ordered by descending value and, among equal values, ascending index.
///
/// The set is PyTorch's; the order within it is Turbine's (routing only uses the set: the
/// experts' outputs are accumulated in ascending expert id whatever the slot order).
///
/// # Panics
/// If `k` exceeds `values.len()` or `values.len()` exceeds `u32::MAX`.
pub fn torch_topk(values: &[f32], k: usize) -> Vec<usize> {
    assert!(k <= values.len(), "top-{k} of {} values", values.len());
    let n = u32::try_from(values.len()).expect("at most u32::MAX values");
    if k == 0 {
        return Vec::new();
    }
    let mut v: Vec<Elem> = values.iter().copied().zip(0..n).collect();
    if k * 64 <= values.len() {
        // std::partial_sort: heap select, then a heap sort of the selection (order only).
        heap_select(&mut v, 0, k, values.len());
    } else {
        nth_element(&mut v, k - 1);
    }
    let mut chosen: Vec<Elem> = v[..k].to_vec();
    chosen.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    chosen.into_iter().map(|e| e.1 as usize).collect()
}

#[cfg(test)]
mod tests {
    use super::torch_topk;

    /// The cases of `torch_topk_fixture.txt`: `k;values;selected indices ascending`.
    fn fixture() -> Vec<(usize, Vec<f32>, Vec<usize>)> {
        include_str!("torch_topk_fixture.txt")
            .lines()
            .filter(|l| !l.is_empty())
            .map(|line| {
                let mut parts = line.split(';');
                let mut next = || parts.next().expect("three fields");
                let k = next().parse().expect("k");
                let values = next()
                    .split(' ')
                    .map(|v| v.parse().expect("value"))
                    .collect();
                let selected = next()
                    .split(' ')
                    .map(|v| v.parse().expect("index"))
                    .collect();
                (k, values, selected)
            })
            .collect()
    }

    /// Every fixture row selects exactly the set `torch.topk` 2.9.0 selected. Breaks if the
    /// selection falls back to "lower index first" (most rows straddle a tie at the k-th
    /// place) or if either libstdc++ path (introselect for 64/8, heap select for 256/4) drifts.
    #[test]
    fn selects_the_same_set_as_torch_topk() {
        let cases = fixture();
        assert!(cases.len() >= 200, "{} cases", cases.len());
        let mut lower_id_differs = 0;
        let mut heap_path = 0;
        for (i, (k, values, want)) in cases.iter().enumerate() {
            let got = torch_topk(values, *k);
            assert_eq!(got.len(), *k);
            let mut set = got.clone();
            set.sort_unstable();
            assert_eq!(&set, want, "case {i}: k={k} values={values:?}");
            // Descending value, ascending index among ties.
            assert!(
                got.windows(2)
                    .all(|w| values[w[0]] > values[w[1]]
                        || values[w[0]] == values[w[1]] && w[0] < w[1]),
                "case {i}: order {got:?}"
            );
            let mut lower: Vec<usize> = (0..values.len()).collect();
            lower.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
            lower.truncate(*k);
            lower.sort_unstable();
            lower_id_differs += usize::from(lower != *want);
            heap_path += usize::from(k * 64 <= values.len());
        }
        assert!(
            lower_id_differs > cases.len() / 4,
            "the fixture must exercise ties: only {lower_id_differs} rows differ from lower-id"
        );
        assert!(heap_path > 0, "no partial_sort row");
    }

    /// Without ties the selection is the `k` largest, NaN counting as the largest (PyTorch's
    /// NaN-first comparator), in descending order.
    #[test]
    fn distinct_values_select_the_largest() {
        let values = [0.5, 3.0, -1.0, 2.0, f32::NAN, 1.0];
        assert_eq!(torch_topk(&values, 3), [4, 1, 3]);
        assert_eq!(torch_topk(&values[..4], 2), [1, 3]);
        assert_eq!(torch_topk(&values, 0), Vec::<usize>::new());
        let wide: Vec<f32> = (0..256).map(|i| ((i * 37) % 256) as f32).collect();
        // 256 values, k 4: the partial_sort path.
        let got = torch_topk(&wide, 4);
        assert_eq!(
            got.iter().map(|&i| wide[i]).collect::<Vec<_>>(),
            [255.0, 254.0, 253.0, 252.0]
        );
    }
}
