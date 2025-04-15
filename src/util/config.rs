use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Config {
  pub task: String,
  pub onnx: OnnxConfig,
  pub ptau: PtauConfig,
  pub sf: ScaleFactorConfig,
  pub prover: ProverConfig,
  pub verifier: VerifierConfig,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct PtauConfig {
  pub ptau_path: String,
  pub pow_len_log: usize,
  pub loaded_pow_len_log: usize,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct OnnxConfig {
  pub model_path: String,
  pub input_path: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ScaleFactorConfig {
  pub scale_factor_log: usize,
  pub cq_range_log: usize,
  pub cq_range_lower_log: usize,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ProverConfig {
  pub model_path: String,
  pub setup_path: String,
  pub enc_model_path: String,
  pub enc_input_path: String,
  pub enc_output_path: String,
  pub proof_path: String,
  pub acc_proof_path: String,
  pub final_proof_path: String,
  pub enable_layer_setup: bool,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct VerifierConfig {
  pub enc_model_path: String,
  pub enc_input_path: String,
  pub enc_output_path: String,
  pub proof_path: String,
}
