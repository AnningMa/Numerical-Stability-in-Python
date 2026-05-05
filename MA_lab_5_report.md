Name: Anning Ma
Student Number: 22404528
Github Link:

# Basic Understanding Questions
1. Why can computers not represent all real numbers exactly?
Computers manipulate numbers using a finite amount of memory, whereas the set of real numbers is infinite.
2.  What is the role of the mantissa in a floating-point number?
The mantissa encodes the significant digits of a floating-point number. In normalised form, it represents the fractional part after an implicit leading 1.
3. What happens when a number is too large to be represented?
When a number exceeds the maximum representable value, overflow occurs and the result is set to infinity (±∞).
4. Why is float16 more unstable than float32?
Float16 is less stable than float32 for two reasons. First, it uses fewer bits overall, so its representation is coarser. Second, it reduces both the mantissa and the exponent compared to float32. The smaller exponent range makes overflow and underflow significantly more likely.
5. Why do machine learning models often not use float64? 
Machine learning systems are built around a trade-off between computational speed, memory usage, and numerical precision. Float64 offers higher precision and a wider dynamic range, but at the cost of greater memory consumption and slower operations. Float32 has therefore long been the default choice, as it provides a good compromise between precision and efficiency.

6. Why does reducing numerical precision (e.g. from float32 to float16) improve both speed and memory usage, and why can this become problematic for training?
Storing model parameters in float16 requires half the memory of float32, and lower-precision arithmetic runs faster on modern GPUs that provide hardware support for reduced-precision operations.
However, this can destabilise training. Many operations in machine learning (such as gradient accumulation and normalisation) are sensitive to numerical errors. Reducing precision can cause unstable training dynamics, vanishing gradients, or incorrect parameter updates.

7. Explain why bfloat16 is generally more numerically stable than float16, even though both use 16 bits.
Unlike float16, which reduces both the mantissa and the exponent relative to float32, bfloat16 retains the same 8-bit exponent while only reducing the mantissa. This gives bfloat16 the same wide dynamic range as float32, making it far less prone to overflow and underflow than float16.

8. In a Hugging Face workflow, why is it often safe to use lower precision for inference but more risky during training? 
Training is more sensitive to numerical errors because it involves gradient accumulation and repeated weight updates, where small errors compound over time.  By contrast, inference is a single forward pass with no parameter updates, so minor numerical imprecision rarely affects the final output. This makes it safe to use lower precision during inference, reducing both memory usage and latency.

# Numerical Stability Experiments
9. Explain why the two expressions give different results, even though addition is associative over real numbers.
```
import numpy as np

a = np.float32(1e8) 
b = np.float32(-1e8) 
c = np.float32(1)

print((a + b) + c)  #1
print(a + (b + c)) #0
```
Floating-point arithmetic breaks the associative rule because of rounding.
1)  `(a+b)+c` gives 1.0 because `1e8 + (-1e8) = 0.0` exactly, and `0.0 + 1.0 = 1.0`.
2) `a+(b+c)` gives 0 because:
    - `(-1e8) + 1.0 = -99999999.0`. But float32 only has ~7 significant decimal digits, so -99999999.0 rounds to -100000000.0 (-1e8).
    -  `1e8 + (-1e8) = 0`

10. Why is the comparison false in the following code?
```
x = 0.1+0.2
y = 0.3
print(x == y) #False
print(x, y) #0.30000000000000004 0.3
```
1. Because 0.1, 0.2 and 0.3 have no exact finite representation in binary. 
0.1 is stored as `0.1000000000000000055511151...`, 
0.2 is stored as `0.2000000000000000111022302...`, 
so the sum is `0.30000000000000004...`
2. 0.3 is stored as `0.29999999999999...`, Python's print displays the shortest decimal string that uniquely identifies each float. since x and y have different stored values, x must be displayed with more digits to distinguish it from 0.3.

11. What happens here? Identify overflow and underflow.
```
x = np.exp(1000)
y = np.exp(-1000)
print(x) #inf
print(y) #0.0
```
The default type is float64. `x = np.exp(1000)` causes overflow because e^1000 far exceeds float64's maximum representable value, so the result is stored as `inf`. 
NumPy raises a RuntimeWarning to signal this.
`y = np.exp(-1000)` causes underflow because e^-1000 is a positive number extremely close to zero, too small for float64 to represent, so it is rounded down to `0.0`.

12. Why is the following code numerically unstable?
```
x = 1e8
y = np.sqrt(x + 1) - np.sqrt(x)
print(y) 
```
When x = 1e8, sqrt(1e8 + 1) and sqrt(1e8) are nearly identical in value. And when sqrt(1e8 + 1) -  sqrt(1e8), the significant digits are cancelled out:  `10000.00005... - 10000.00000...`, leaving only the last few digits dominated by rounding errors(0.00005...). The result therefore has very few reliable significant digits.

13. What goes wrong?
```
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

x = np.array([1000, 1001, 1002])
print(softmax(x))
```
The function outputs `[nan nan nan]` due to overflow.
`np.exp(1000)`, `np.exp(1001)`, and `np.exp(1002)` all overflow to `inf`, because np.exp overflows to inf for inputs above ~709 in float64 (since e^709 ≈ 8.2 × 10^307, which is close to float64's maximum). Dividing inf by inf (inf / (inf + inf + inf)) produces nan.

14. Compute softmax by hand for a small vector

```
def stable_softmax(x):
    z = x - np.max(x)
    return np.exp(z) / np.sum(np.exp(z))
```
If the vecctor is x=[-1,2], softmax(x)=[0.0474, 0.9526]. There isn't overflow or underflow.
If the vector is x = [-1000,-1001], the result by hand is about [0.7311,0.2689]

15. Compare naive vs stable vs PyTorch
x = [1000.0, 1001.0, 1002.0]
1) Naive:
np.exp(1000), np.exp(1001) and np.exp(1002) are stored as inf(overflow), so the final result is [nan, nan]
2) Stable:
Subtracting the maximum value (1002) shifts the inputs to [-2, -1, 0], so the largest exp value is exp(0) = 1, which avoids overflow entirely. 
the result: [0.0900, 0.2447, 0.6652]
3) Pytorch
torch.softmax applies the same max-subtraction trick as stable softmax, so it produces the result [0.0900, 0.2447, 0.6652] 

16. Why does subtracting the maximum fix the problem?
1) softmax is invariant to constant shifts.
2) By choosing c=max⁡(x), the largest input becomes 0, and all others become negative — so every exp value stays in the range (0, 1], which can never overflow.

# Report:Numerical Precision, Quantisation, and Distance Structure in wav2vec Representations

## Introduction

This report is based on a reproducible DVC pipeline that extracts word-level 
features from a wav2vec model and compares cosine distance matrices across 
four numerical precisions: float64 (baseline), float32, float16, and a 8-bit quantization scheme. The pipeline consists of the following stages:

- parse-corpus: Extracts and filters words from the raw data.

- extract_features: Loads a pretrained wav2vec model, extracts frame-level 
  representations for each word's audio segment, aggregates them into a 
  word-level vector via mean pooling, and saves the result in float64 
  as the baseline.

- convert_1: Converts the float64 representations to float32 and float16.

- convert_2: Implements a per-row 8-bit quantization scheme, where each 
  embedding vector has its own scale and zero_point computed independently.

- convert_3: Implements a per-tensor 8-bit quantization scheme, where a single 
  shared scale is applied across the entire matrix, mapping values to [-127, 127]. 
  
  Results are compared against the per-row scheme.

- compute_distance: Computes the following for each precision:
  1) Pairwise cosine distance matrix across all words
  2) Computation time for each distance matrix
  3) Mean intra-speaker distance, mean inter-speaker distance, 
     and their ratio

- visualize: Visualizes all of the above results.

## 8-bit Quantization
8-bit quantization maps floating-point values (e.g. float64) to 256 integer 
values (int8) using the formula:

$$x_quantized = round(x / scale + zero_point)$$

The original values are reconstructed via the inverse formula: $x_reconstructed = (x_quantized - zero_point) × scale$

The choice of quantization method comes down to how scale and zero_point are 
determined. Here I compared two approaches:
1) Per-tensor: a single scale is shared across the entire matrix, with zero_point = 0, scale = max(|all values|) / 127

2) Per-row: each embedding vector has its own scale. 
$scale = max(|values in that row|) / 127$

As shown in the comparison table below, the per-row scheme achieves a 3× lower 
mean error than the per-tensor scheme. The larger error in the per-tensor approach stems from using a single global scale, which fails to capture the dynamic range of individual rows and therefore leads to greater precision loss.

| Metric          | Per-tensor       | Per-row          | Ratio (per-tensor / per-row) |
|-----------------|------------------|------------------|------------------------------|
| Max \|error\|   | 1.221 × 10⁻²     | 9.300 × 10⁻³     | 1.3× worse                   |
| Mean \|error\|  | 6.107 × 10⁻³     | 2.060 × 10⁻³     | 3.0× worse                   |

## Comparison
1) The table below summarises the storage size, maximum error, and mean error across four numerical formats: float64, float32, float16, and int8.

| Format  | Size      | Compression | Max \|error\|  | Mean \|error\| |
|---------|-----------|-------------|----------------|----------------|
| float64 | 40.13 MB  | ×1.0 (ref)  | —              | —              |
| float32 | 20.07 MB  | ×2.0        | 0.000000e+00   | 0.000000e+00   |
| float16 | 10.03 MB  | ×4.0        | 9.668 × 10⁻⁴   | 3.555 × 10⁻⁵   |
| uint8    | 5.08 MB*  | ×7.9        | 9.300 × 10⁻³   | 2.060 × 10⁻³   |

The results show a trade-off between compression and precision. Float32 achieves ×2.0 compression with zero error, making it a lossless reduction from float64. Float16 compresses ×4.0 with mean error 3.555 × 10⁻⁵, remaining a reliable choice for most use cases. uint8 offers the highest compression but at the cost of significantly higher error (mean: 2.060 × 10⁻³). This suggests that while int8 is effective for memory-constrained settings, it introduces non-trivial quantization noise that may affect downstream tasks such as speaker discrimination based on cosine distance.

2)  The table below summarises the different compute time across four numerical formats:
| Format  | Compute Time (s) | File Size (MB) |
|---------|-----------------|----------------|
| float64 | 1.128           | 40.13          |
| float32 | 0.534           | 20.07          |
| float16 | 0.593           | 10.03          |
| int8    | 1.050           | 5.08           |

Float32 achieves the fastest computation (0.534s), roughly 2× faster than float64. Float16 further reduces storage to 10.03 MB but offers no additional speed benefit over float32. Int8 achieves the smallest footprint (5.08 MB) but is slow (1.050s), due to the overhead of per-row quantization and dequantization. 

Overall, float32 offers the best speed-to-size trade-off in this pipeline.

3) Now we compare the cosine distance matrix statistics across formats, including mean intra-speaker distance, mean inter-speaker distance, and their ratio.

Float32 and float16 produce identical intra/inter/ratio values to float64, indicating that reducing precision has no measurable impact on speaker discrimination for this dataset. Int8 introduces a negligible deviation, which is unlikely to affect any downstream task.

| Format  | Intra-speaker | Inter-speaker | Ratio  |
|----------|---------------|---------------|--------|
| float64  | 0.344420      | 0.414566      | 1.2037 |
| float32  | 0.344420      | 0.414566      | 1.2037 |
| float16  | 0.344420      | 0.414566      | 1.2037 |
| int8       | 0.344467      | 0.414607      | 1.2036 |

## Visualization
Consin Distance Distribution:
![alt text](image.png)
Inter-speaker VS Intra-speaker Mean Distance:
![alt text](image-1.png)
Efficiency:
![alt text](image-2.png)

The wav2vec2 model operates in float32 within PyTorch, so float64 and float32 produce identical distance matrices. For float16, the rounding errors are small enough that they only appear beyond the 7th decimal place, leaving the distances unchanged after rounding to 6 decimal places. Across all formats, the mean inter-speaker distance consistently exceeds the mean intra-speaker distance, with no change in relative ordering and a negligible change in ratio.

## Conclusion
The results demonstrate that reducing numerical precision from float64 to float32, float16, and 8-bit quantisation inevitably introduces approximation errors in absolute values. However, the ratio between intra- and inter-speaker distances remains highly consistent across all precision levels, suggesting that lower precision perturbs individual values without fundamentally altering the structure of the representation space. In practice, lower precision formats can significantly reduce storage and computational costs, offering a strong trade-off between efficiency and numerical fidelity, and can therefore be safely adopted for large-scale speech analysis.

But this conclusion should not be generalised without caution. Since wav2vec2 is natively trained in float32, the precision ceiling is already at float32, meaning float16 introduces no additional loss for this model. Furthermore, speaker discrimination is a relatively coarse-grained task; for finer-grained tasks, the impact of precision reduction may be more important.

