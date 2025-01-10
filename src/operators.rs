use crate::tensor::Tensor;

// 张量操作相关的模块，实现了一系列神经网络中常用的操作符

// gather函数用于从二维表格(table)中根据索引(indices)获取对应的行向量
// 参数说明:
// - y: 输出张量,用于存储获取的向量
// - indices: 索引张量,包含要获取的行号
// - table: 输入表格张量,包含源数据
// mut: 表示变量是可变的(mutable),可以修改其值。如果不加mut则变量默认是不可变的(immutable)
// f32: 32位浮点数类型,相当于其他语言中的float
// u32: 32位unsigned(无符号)整数类型,只能表示非负整数
// &: 引用符号,表示借用变量而不获取其所有权。&mut表示可变引用,允许修改引用的值
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    // pub 是 Rust 中的访问修饰符,表示这个函数/结构体/模块是公开的,可以被其他模块访问
    // 如果不加 pub,默认是私有的,只能在当前模块内部使用
    // 获取索引张量的长度
    let length = indices.size();
    // 获取输入表格的形状
    let table_shape = table.shape();
    // 确保输入表格是二维的
    assert!(table_shape.len() == 2);
    // 获取表格的列数(每行向量的维度)
    let dim = table_shape[1];
    // 确保输出张量的大小等于索引数量乘以向量维度
    assert!(y.size() == length * dim);
    // 遍历每个索引
    for i in 0..length {
        // 根据索引获取源表格中对应行的切片
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        // 获取输出张量对应位置的可变切片
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        // 将源数据复制到目标位置
        dst.copy_from_slice(src);
    }
}

// RoPE: 旋转位置编码
// 通过对输入进行旋转变换来编码位置信息
// 参数:
// - y: 需要添加位置编码的张量
// - start_pos: 起始位置
// - theta: 旋转角度参数
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    // 获取张量的形状
    let shape = y.shape();
    // 确保张量是三维的
    assert!(shape.len() == 3);
    // 获取序列长度、头数和每个头的维度
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    // 获取张量的数据的可变引用
    let data = unsafe { y.data_mut() };
    // 遍历序列中的每个token
    for tok in 0..seq_len {
        // 计算当前token的位置
        let pos = start_pos + tok;
        // 遍历每个头
        for head in 0..n_heads {
            // 遍历每个头的一半维度
            for i in 0..d / 2 {
                // 获取当前维度的两个值
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                // 计算频率
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                // 计算sin和cos值
                let (sin, cos) = freq.sin_cos();
                // 更新张量数据
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// masked_softmax: 带掩码的softmax操作
// 计算公式: softmax(x) = exp(x - max) / sum(exp(x - max))
// 对输入进行掩码处理后计算softmax
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

// rms_norm: 均方根层归一化
// 参数:
// - y: 输出张量
// - x: 输入张量
// - w: 权重张量
// - epsilon: 数值稳定性参数
pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    // 获取输入张量的形状
    let shape = x.shape();
    // 确保输入张量和输出张量的形状相同
    assert!(shape == y.shape());
    // 确保权重张量的大小等于输入张量的最后一个维度。
    assert!(w.size() == shape[shape.len() - 1]);
    // 获取输入张量的最后一个维度
    let n = shape[shape.len() - 1];
    // 计算每个batch的大小
    let batch_size = x.size() / n;
    // 获取输入张量、权重张量和输出张量的数据
    let x_data = x.data();
    let w_data = w.data();
    let y_data = unsafe { y.data_mut() };
    // 遍历每个batch
    for i in 0..batch_size {
        // 计算当前batch的偏移量
        let offset = i * n;
        // 计算输入张量的平方和
        let sum_of_squares: f32 = (0..n).map(|j| x_data[offset + j].powi(2)).sum();
        // 计算均方根
        let norm = (sum_of_squares / n as f32 + epsilon).sqrt();
        // 计算输出张量
        for j in 0..n {
            y_data[offset + j] = w_data[j] * x_data[offset + j] / norm;
        }
    }
}
    

// swiglu: SwiGLU激活函数
// 计算 y = silu(x) * y
// 这是一个按元素进行的操作
fn silu(x: f32) -> f32 {
    let sigmoid = 1.0 / (1.0 + (-x).exp());
    sigmoid * x
}

pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert!(len == x.size());

    let y_data = unsafe { y.data_mut() };
    let x_data = x.data();

    for i in 0..len {
        y_data[i] = silu(x_data[i]) * y_data[i];
    }
}

// matmul_transb: 矩阵乘法（第二个矩阵转置）
// 计算: C = beta * C + alpha * A @ B^T
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    // 获取输入张量的形状
    let a_shape = a.shape();
    let b_shape = b.shape();
    let c_shape = c.shape();
    
    // 确保输入张量是二维的
    assert!(a_shape.len() == 2 && b_shape.len() == 2 && c_shape.len() == 2);
    
    // 确保输入张量的形状符合矩阵乘法的要求
    assert!(a_shape[1] == b_shape[1]);
    assert!(a_shape[0] == c_shape[0]);
    assert!(b_shape[0] == c_shape[1]);
    
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[0];
    
    let a_data = a.data();
    let b_data = b.data();
    let c_data = unsafe { c.data_mut() };
    
    // 计算 C = alpha * A * B^T + beta * C
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a_data[i * k + p] * b_data[j * k + p];
            }
            c_data[i * n + j] = alpha * sum + beta * c_data[i * n + j];
        }
    }
}

// dot: 计算两个张量的点积
// 将输入张量视为向量进行内积计算
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// random_sample: 从概率分布中采样
// 参数:
// - x: 概率分布张量
// - top_p: nucleus sampling参数
// - top_k: top-k采样参数
// - temperature: 温度参数，控制分布的平滑程度
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    swiglu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}
