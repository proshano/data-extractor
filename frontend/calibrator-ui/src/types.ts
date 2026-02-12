export type Feature = {
  name: string;
  description: string;
  missing_value_rule: string;
  prompt: string;
  allowed_values?: string[];
  type_hint?: string;
};

export type ReasoningMode = "direct" | "plan_then_extract" | "react_style" | "custom";

export type JudgeConfig = {
  enabled: boolean;
  model: string;
  instructions: string;
  acceptance_threshold: number;
};

export type ExperimentProfile = {
  id: string;
  name: string;
  system_instructions: string;
  extraction_instructions: string;
  reasoning_mode: ReasoningMode;
  reasoning_instructions: string;
  output_instructions: string;
  judge: JudgeConfig;
};

export type CsvPreviewRow = {
  row_number: number;
  values: Record<string, string>;
};

export type CsvLoadResponse = {
  source: string;
  columns: string[];
  row_count: number;
  preview: CsvPreviewRow[];
  encoding: string;
  delimiter: string;
  inferred_id_column: string;
  inferred_report_column: string;
};

export type ModelListResponse = {
  llama_url: string;
  models: string[];
};

export type LocalModelListResponse = {
  models: string[];
  timed_out: boolean;
  binary_path: string;
};

export type HuggingFaceGgufFile = {
  file_name: string;
  size_bytes: number | null;
  size_gb: number | null;
};

export type HuggingFaceGgufModel = {
  repo_id: string;
  downloads: number;
  likes: number;
  last_modified: string;
  gguf_files: HuggingFaceGgufFile[];
};

export type HfGgufSearchResponse = {
  models: HuggingFaceGgufModel[];
};

export type HfGgufFilesResponse = {
  repo_id: string;
  files: HuggingFaceGgufFile[];
};

export type HfGgufDownloadResponse = {
  repo_id: string;
  file_name: string;
  destination_dir: string;
  downloaded_path: string;
};

export type LlamaServerStatusResponse = {
  process_running: boolean;
  managed_model_path: string;
  managed_port: number | null;
  managed_ctx_size: number | null;
  started_at_unix: number | null;
  changed?: boolean;
  llama_url: string;
  binary_path: string;
  reachable: boolean;
  server_models: string[];
  connect_error: string;
  logs_tail: string[];
};

export type SessionState = {
  version: number;
  csv_path: string;
  csv_cache?: CsvLoadResponse | null;
  schema_path: string;
  llama_url: string;
  temperature: number;
  max_retries: number;
  id_column: string;
  report_column: string;
  sample_index: number;
  server_model: string;
  features: Feature[];
  experiment_profiles: ExperimentProfile[];
  active_experiment_id: string;
  active_experiment_ids: string[];
};

export type JudgeDecision = "accepted" | "rejected" | "uncertain" | "judge_error";

export type JudgeResult = {
  status: JudgeDecision;
  verdict: string;
  score: number | null;
  rationale: string;
  raw_response: string;
  error: string;
  model: string;
  duration_ms: number;
};

export type FeatureTestResult = {
  feature_name: string;
  status: "ok" | "llm_error" | "parse_error";
  value: string;
  parsed: Record<string, unknown> | null;
  raw_response: string;
  error: string;
  model: string;
  duration_ms: number;
  run_index?: number;
  report_index?: number;
  row_number?: number | null;
  study_id?: string;
  experiment_id?: string;
  experiment_name?: string;
  judge_result?: JudgeResult | null;
};

export type ReportTestResult = {
  run_index?: number;
  report_index: number;
  row_number: number | null;
  study_id?: string;
  report_text: string;
  experiment_id?: string;
  experiment_name?: string;
  results: FeatureTestResult[];
};

export type JobResponse = {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  total: number;
  reports_total?: number;
  reports_completed?: number;
  active_report_index?: number | null;
  completed: number;
  progress_percent: number;
  created_at?: number;
  updated_at?: number;
  started_at?: number | null;
  finished_at?: number | null;
  results: FeatureTestResult[];
  report_results?: ReportTestResult[];
  last_result?: FeatureTestResult | null;
  error: string;
  cancel_requested: boolean;
};

export type ExtractionJobResponse = {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  total_rows: number | null;
  processed_rows: number;
  ok_rows: number;
  error_rows: number;
  elapsed_seconds: number;
  rows_per_minute: number;
  progress_percent: number | null;
  last_source_row_number: number | null;
  id_column: string;
  report_column: string;
  output_csv_path: string;
  resume_mode?: "fresh" | "resumed";
  processed_rows_at_start?: number;
  created_at?: number;
  updated_at?: number;
  started_at?: number | null;
  finished_at?: number | null;
  error: string;
  cancel_requested: boolean;
};
