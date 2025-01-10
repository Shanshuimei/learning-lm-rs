use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}


impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // 定义一个闭包函数get_tensor，用于从safetensor中获取张量
        let get_tensor = |name: &str| -> Tensor<f32> {
            // 从safetensor中获取名为name的张量
            let tensor = safetensor.tensor(name).unwrap();
            // 获取张量的形状并转换为Vec
            let shape = tensor.shape().to_vec();
            // 获取张量的数据
            let raw_data = tensor.data();
            // 将数据从u8类型转换为f32类型
            let data: Vec<f32> = raw_data
                // 按照f32类型的字节大小进行精确分块
                .chunks_exact(std::mem::size_of::<f32>())
                // 将每个分块转换为f32类型
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                // 收集所有转换后的f32数据到一个向量中
                .collect();
            // 创建一个新的Tensor并返回
            Tensor::new(data, &shape)
        };

        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"),
            // 使用map函数对0到config.num_hidden_layers范围内的每一个值进行操作
            rms_att_w: (0..config.num_hidden_layers)
                // map函数会对范围内的每一个值应用闭包函数
                // 闭包函数的参数是i，表示当前的层数
                // format!宏用于生成一个字符串，字符串中包含当前层数i
                // get_tensor函数根据生成的字符串从safetensor中获取对应的张量
                .map(|i| get_tensor(&format!("model.layers.{}.input_layernorm.weight", i)))
                // collect函数将map函数生成的迭代器收集成一个向量
                .collect(),
            wq: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", i)))
                .collect(),
            wk: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", i)))
                .collect(),
            wv: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", i)))
                .collect(),
            wo: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", i)))
                .collect(),
            rms_ffn_w: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.post_attention_layernorm.weight", i)))
                .collect(),
            w_up: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", i)))
                .collect(),
            w_gate: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", i)))
                .collect(),
            w_down: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", i)))
                .collect(),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}